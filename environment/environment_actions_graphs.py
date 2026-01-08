# environment_actions_graph.py
# how to implement history list here 

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from core.abstract import BaseTrajectory
from core.utils import softmax
import numpy as np
from environment.flowsheet_simulation_graph import FlowsheetSimulationGraph
import copy, torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import traceback 
from experts.experts import DistillationColumn, Decanter, Mixer, Split, Recycler, AddSolvent, EdgeFlowExpert, OpenStreamExpert, FlowExpert



class FlowsheetDesign:

    """
    Graph-based action environment:

      Level 0: choose OPEN stream (or index 0 = TERMINATE)
      Level 1: choose UNIT type 
      Level 2: choose parameters for the following UNITS <distillation_column, add_solvent, decanter, split, recycle, mixer>
      Level 3: choose further parameters for <add_solvent, mixer>

    After an action completes successfully:
      - placement/wiring happens (add_unit / add_recycle)
      - simulate() runs
      - NPV is recomputed
      - state resets to level 0

    On any failure:
      - we roll back partial changes (including recycle edges)
      - reset to level 0 (no loops)
    """

    def __init__(self, random_instance: Dict[str, Any], gen_config, env_config):
        self.gen_config = gen_config
        self.env_config = env_config 
        self.sim = FlowsheetSimulationGraph(random_instance, self.env_config)
        self.num_units = len(self.env_config.unit_types)

        # add feed nodes
        self.sim.feed_nodes = []
        for feed in random_instance["list_feed_streams"]:
            self.sim.feed_nodes.append(self.sim.add_feed(feed))

        # params variables 
        self.DF_distillation: float = None 
        self.split_ratio: float = None 

        # current action state
        self.level = 0 
        self.chosen_open_stream: Optional[Tuple[int, str]] = None    # (node_id, label)
        self.chosen_unit: Optional[Tuple[int, str]] = None                 # index in units_map_indices_type
        self.pending_params: Optional[Dict] = {}  # params for the chosen unit  
        self.second_open_stream_dest_node: Optional[int] = None     # for mixer
        self.second_open_stream: Optional[Tuple[int, str]] = None      
        self.recycle_dest_unit: Optional[int] = None                  # for recycle
        self.recycler_mask: torch.Tensor= None # for attentive mask for recycler expert
        self.mixer_mask: torch.Tensor = None # for attentive mask for mixer expert 
        self.padded_open_stream_mask: torch.Tensor = None 
        self.padded_recycler_mask: torch.Tensor = None
        self.padded_mixer_mask: torch.Tensor = None
        self.valid_nodes: torch.Tensor = None 

        # counters (limits)
        self.counts = {
            "distillation_column": 0,
            "decanter": 0,
            "split": 0,
            "mixer": 0,
            "recycle": 0,
            "add_solvent": 0,
        }

        # define all unit experts to get embeddings 
        self.unit_experts = {
            "distillation_column": DistillationColumn(self.gen_config), 
            "decanter": Decanter(self.gen_config),   
            "mixer": Mixer(self.gen_config),
            "split": Split(self.gen_config),
            "recycle": Recycler(self.gen_config), 
            "add_solvent": AddSolvent(self.gen_config, self.env_config),      
            "flow_expert": FlowExpert(self.gen_config, self.env_config)
            }

        self.edge_expert = EdgeFlowExpert(self.gen_config, self.unit_experts["flow_expert"])
        self.open_stream_expert = OpenStreamExpert(self.gen_config, self.env_config, self.unit_experts["flow_expert"])
        self.total_units_placed = 0

        # history stores a list of all action taken across 4 levels. whenever an action is not taken at a certain level (SKIP_action)
        # it is assigned number 901 
        self.SKIP_ACTION = 901
        self.history: List[int] = [] 
        
        self.finished_design = False # True only when termination criteria has been reached 
        self.current_action_mask: Optional[np.array] = None # The action mask indicates before each action what is feasible at the current level.

        # initial simulate to populate open streams/NPV
        self.sim.simulate()
        self.current_state = self.get_current_state()
        self.get_feasible_actions()

    def get_current_state(self) -> Dict[str, Any]:
        
        state = {
            "current_level": self.level,
            "open_streams": self._enumerate_open_streams(),
            "chosen_open_stream": self.chosen_open_stream,
            "chosen_unit": self.chosen_unit,
            "pending_params": self.pending_params, 
            "npv_raw": self.sim.current_net_present_value,
            "npv_norm": self.sim.current_net_present_value_normed,
            "completed_design": self.finished_design, 
            "second_open_stream_dest_node": self.second_open_stream_dest_node,
            "second_open_stream": self.second_open_stream,
            "recycle_dest_unit": self.recycle_dest_unit,
            "recycler_mask": self.recycler_mask, 
            "mixer_mask": self.mixer_mask, 
            "current_action_mask": self.current_action_mask
                            }
        return state

    def get_feasible_actions(self) -> np.ndarray:

        """
        Return a 0/1 vector mask feasible actions for the current level:

        Level 0: indices enumerate all open streams, + 1 for "terminate"
        Level 1: indices enumerate unit choices (0...num_units-1) 
        Level 2: select parameters values for the following units (distillation column: DF value (100), split: split_ratio (100), 
        mixer: 2nd open node (N = num of nodes within the graph), add_solvent: select a compound (from 5 options)
        recycler: destination node (N = num of nodes within the graph))
        Level 3: select amount for selected compound for add_solvent (100) or select open stream for the selected node for mixer 

        0: the action is masked. 1 = action is allowed

        """
        if self.level == 0:
            open_streams = self._enumerate_open_streams()
            mask = np.ones(len(open_streams) + 1, dtype=int) #check if this is valid 
            
            # masking condition for lvl 0 (terminate)
            if self.finished_design == True:   
                mask[1:len(open_streams)] = 0
            else:
                mask[0] = 0  #index 0 for termimate 

            # enable all available stream slots (everything cis available)
            self.current_action_mask = mask
            self.current_state["current_action_mask"] = self.current_action_mask 
            return mask   

        # select units now 
        elif self.level == 1:
            unit_params_mask = np.zeros(self.env_config.num_units, dtype=int)
            for idx, unit_name in enumerate(self.env_config.units_map_indices_type):
                # allow only avail units
                avail_unit = self._unit_available(unit_name, idx)
                if unit_name not in ["mixer", "recycle"] and avail_unit:
                    unit_params_mask[idx] = 1
                if unit_name == "mixer" and avail_unit:
                    # need at least 2 open streams
                    if len(self._enumerate_open_streams()) < 2:
                        continue
                    else:
                        src_node, _ = self.current_state["chosen_open_stream"]
                        candidates = self._enumerate_open_streams_excluding(exclude = self.current_state["chosen_open_stream"])
                        candidate_nodes = sorted({i for i, _ in candidates if i != src_node})
                        if len(candidate_nodes) < 1:
                            continue 
                        else:
                            unit_params_mask[idx] = 1
                if unit_name == "recycle" and avail_unit:
                    # need a chosen stream (comes from level 0) and at least one eligible dest
                    eligible_recycle_destinations = self._eligible_recycle_destinations()
                    if self.current_state["chosen_open_stream"] is None or len(eligible_recycle_destinations) == 0:
                        continue
                    else:
                        unit_params_mask[idx] = 1

            self.current_action_mask = unit_params_mask
            return unit_params_mask

        elif self.level == 2:
            open_streams = self.current_state["open_streams"]
            _, chosen_unit_name = self.current_state["chosen_unit"]

            if chosen_unit_name in ["distillation_column", "split"]:
                params_mask = np.ones(100, dtype=int) # what constraints can be present here?

            elif chosen_unit_name == "add_solvent":
                params_mask = np.ones(5, dtype= int) # what constraints can be present here?

            elif chosen_unit_name == "recycle":
                node_ids = list(self.sim.graph.nodes)
                source_node, _ = self.current_state["chosen_open_stream"]
                id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
                params_mask = np.zeros(len(node_ids), dtype=int) # should recycle be better open_streams??
                recycle_dests = self._eligible_recycle_destinations()
                for i in recycle_dests:
                    if i != source_node:
                        idx = id_to_idx[i]
                        params_mask[idx] = 1 

            elif chosen_unit_name == "mixer":
                candidates = self._enumerate_open_streams_excluding(exclude = self.current_state["chosen_open_stream"])
                node_ids = list(self.sim.graph.nodes)
                id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
                source_node, _ = self.current_state["chosen_open_stream"]
                params_mask = np.zeros(len(node_ids), dtype=int) # number of nodes present in the graph
                for i, _ in candidates:
                    if i != source_node:
                        idx = id_to_idx[i]
                        params_mask[idx] = 1 
            
            self.current_action_mask = params_mask
            return params_mask

        elif self.level == 3:
            _, unit_name = self.current_state["chosen_unit"]
            if unit_name == "add_solvent":
                params_mask =  np.ones(100, dtype=int)
                self.current_action_mask = params_mask #now decide components for parameters

            if unit_name == "mixer":
                candidates = self._enumerate_open_streams_excluding(exclude = self.current_state["chosen_open_stream"])
                source_node, _ = self.current_state["chosen_open_stream"]
                params_mask = np.zeros(len(candidates), dtype=int)
                for i, _ in candidates:
                    if i != source_node:
                        params_mask[i] = 1

                self.current_action_mask = params_mask # decide which outlet to select for available nodes

            return params_mask
        
        return np.array([], dtype=int)

    def take_action(self, action_index: int) -> Tuple[bool, float, bool]:

        """
        A action index that agent takes across different levels. 
        Actions can be taken across 4 levels in a hierarchical way. 

        Level 0: choose either to terminate design <index 0> or select an open stream 
        Level 1: For a given stream, select a unit type: <distillation_column, add_solvent, decanter, split, recycle, mixer>
        Level 2: Select parameters values for the following units 
         -- distillation column: select a distillation fraction value 
         -- decantor: nothing 
         -- split: select split ratio
         -- recycle: select the destination node, given a source node 
         -- mixer: select the destination node with open streams 
         -- add_solvent: select a compound between 5 global available compounds 
        
        Level 3: Further parameter selection for units add_solvent and mixer 
        -- select amount for selected compound for add_solvent 
        -- select open stream for the selected node for mixer 

        Returns:
          finished_design (bool) = true if termination chosen or max units reached
          reward (float) = current (normalized) NPV after completing an action
          move_worked (bool) = false if placement failed (e.g., PEQ failure / convergence fail)

        """

        assert not self.finished_design, "Taking action on an already terminated design!"
        '''assert self.current_action_mask[action_index] == 0, \
            f"Trying to take action {action_index} on level {self.level}, but it is set to infeasible"'''
        if action_index >= len(self.current_action_mask):
            raise ValueError(f"Invalid action {action_index}, mask size {len(self.current_action_mask)}") 
        
        try:
            action_index = int(action_index)
            self.current_state['current_level'] = self.level 
            self.current_state['current_action_mask'] = self.current_action_mask
            open_streams = self._enumerate_open_streams()
            self.current_state['open_streams'] = open_streams

            if self.level == 0:      
                if action_index == 0:  # Check this later 
                    self.finished_design = True 
                    self.history.extend([action_index] + [self.SKIP_ACTION] * 3) # because terminate do not have lvl1, lvl 2, lvl 3 decisions, we assign "skip_action" as 901
                    self.current_state["completed_design"] = True 
                    self.current_state["npv_norm"] = (self.sim.current_net_present_value_normed or 0.0) 
                    return True, (self.sim.current_net_present_value_normed or 0.0), True
                
                # if not terminate index, then open_stream selected 
                selected_stream = open_streams[action_index - 1], 
                self.current_state['chosen_open_stream'] = selected_stream[0]
                self.level = 1
                self.history.append(action_index)
                self.get_feasible_actions()
                return False, 0.0, True

            elif self.level == 1:
                unit_idx = action_index
                if unit_idx < 0 or unit_idx >= self.env_config.num_units:
                    raise ValueError("Illegal unit index.")
                unit_name = self.env_config.units_map_indices_type[unit_idx]
                if not self._unit_available(unit_name, unit_idx):
                    raise ValueError("Unit not available due to limits or feasibility.")
            
                if unit_name == "distillation_column":
                    # add a new category for DF ratio
                    self.pending_params["distillation_column"] = (int, int) # tuple of index, value of the chosen DF 
                    self.level = 2
                    self.chosen_unit = (unit_idx, unit_name)
                    self.current_state['chosen_unit'] = self.chosen_unit
                    self.history.append(action_index)
                    self.get_feasible_actions()
                    return False, 0.0, True
                
                if unit_name == "split":
                    # add a new category for DF ratio
                    self.pending_params["split"] = (int, int) # tuple of index, value of the chosen ratio
                    self.level = 2
                    self.chosen_unit = (unit_idx, unit_name)
                    self.current_state['chosen_unit'] = self.chosen_unit
                    self.history.append(action_index)
                    self.get_feasible_actions()
                    return False, 0.0, True

                if unit_name == "add_solvent":
                    self.pending_params["add_solvent"] = {}
                    self.level = 2
                    self.chosen_unit = (unit_idx, unit_name)
                    self.current_state['chosen_unit'] = self.chosen_unit
                    self.history.append(action_index)
                    self.get_feasible_actions()
                    return False, 0.0, True

                # recycle (destination selection)
                if unit_name == "recycle":
                    self.pending_params["recycle"] = (int) # index of chosen destination
                    self.level = 2
                    self.chosen_unit = (unit_idx, unit_name)
                    self.current_state['chosen_unit'] = self.chosen_unit
                    self.history.append(action_index)
                    self.get_feasible_actions()
                    return False, 0.0, True
                
                if unit_name == "mixer":
                    self.pending_params['mixer'] = {
                        "index_node": None, 
                        "value_outlet": None, 
                        "tuple_value": None
                    } # tuple of index, value of the 2nd o/p stream
                    self.level = 2
                    self.chosen_unit = (unit_idx, unit_name)
                    self.current_state['chosen_unit'] = self.chosen_unit
                    self.history.append(action_index)
                    self.get_feasible_actions()
                    return False, 0.0, True 

                if unit_name == "decantor":
                    # immediate place (no continuous param, no second stream)
                    self.chosen_unit = (unit_idx, unit_name)
                    self.history.extend([action_index] + [self.SKIP_ACTION] * 2) 
                    self.current_state['chosen_unit'] = self.chosen_unit
                    done, reward, worked = self._complete_action_place_and_simulate()
                    self.level = 0 #back to selecting a new stream or terminating 
                    self.get_feasible_actions()
                    self.current_state["completed_design"] = done 
                    self.current_state["npv_norm"] = reward 
                    return done, reward, worked

            elif self.level == 2:
                # for distillation, decide on which DF ratio value to choose 
                if self._chosen_unit_name() == "distillation_column":
                    if action_index > len (self.env_config.DF_distillation_map):
                        raise ValueError("Distillation fraction value selected more than the permissible limit.")
                    else:
                        self.DF_distillation = self.env_config.DF_distillation_map[action_index]
                        self.pending_params["distillation_column"] = (action_index, self.DF_distillation)
                        self.current_state["pending_params"] = self.pending_params
                        self.history.extend([action_index] + [self.SKIP_ACTION]) 
                        self.level = 0
                        self.get_feasible_actions()
                        done, reward, worked = self._complete_action_place_and_simulate()
                        self.current_state["completed_design"] = done 
                        self.current_state["npv_norm"] = reward 
                        return done, reward, worked

                # for selecting split ratio 
                elif self._chosen_unit_name() == "split":
                    if action_index > len(self.env_config.split_ratio_map):
                        raise ValueError("Split ratio value selected more than the permissible limit.")
                    else:
                        self.split_ratio = self.env_config.split_ratio_map[action_index]
                        self.pending_params["split"] = (action_index, self.split_ratio)
                        self.current_state["pending_params"] = self.pending_params
                        self.history.extend([action_index] + [self.SKIP_ACTION]) 
                        self.level = 0
                        self.get_feasible_actions()
                        done, reward, worked = self._complete_action_place_and_simulate()
                        self.current_state["completed_design"] = done 
                        self.current_state["npv_norm"] = reward 
                        return done, reward, worked
                    
                elif self._chosen_unit_name() == "add_solvent":
                    if action_index > 4:
                        raise ValueError("Invalid selection for thermodynamic components")
                    else:
                        self.chosen_add_solvent_comp = self.env_config.component_names[action_index]
                        self.pending_params["add_solvent"] = {
                            "index_for_comp": action_index,
                            "name_comp": self.chosen_add_solvent_comp,
                            "index_for_amount": None,
                            "amount_value": None,
                        }
                        self.history.append(action_index)
                        self.current_state["pending_params"] = self.pending_params
                        self.level = 3 
                        self.get_feasible_actions()
                        return False, 0.0, True
                        
                # choose destination stream for recycle
                elif self._chosen_unit_name() == "recycle":
                    dests = self._eligible_recycle_destinations()
                    if len(dests) < 0:
                        raise ValueError("No destination stream available for recycle ")
                    else:
                        # select destination and simulate 
                        if action_index in dests:
                            self.recycle_dest_unit = action_index
                        self.pending_params["recycle"] = self.recycle_dest_unit
                        self.current_state["pending_params"] = self.pending_params
                        self.history.extend([action_index] + [self.SKIP_ACTION]) 
                        self.current_state["recycle_dest_unit"] = self.recycle_dest_unit
                        done, reward, worked = self._complete_action_place_and_simulate()
                        self.current_state["completed_design"] = done 
                        self.current_state["npv_norm"] = reward
                        self.level = 0 
                        self.get_feasible_actions()
                        self.current_state['open_streams'] = self._enumerate_open_streams()
                        return done, reward, worked

                # mixer: choose second stream
                elif self._chosen_unit_name() == "mixer":
                    candidates = self._enumerate_open_streams_excluding(exclude = self.current_state["chosen_open_stream"])
                    src_node, _ = self.current_state["chosen_open_stream"]
                    candidate_nodes = sorted({i for i, _ in candidates if i != src_node})
                    if action_index in candidate_nodes:
                        self.second_open_stream_dest_node = action_index 
                    self.pending_params["mixer"]["index_node"] = self.second_open_stream_dest_node
                    self.history.append(action_index) 
                    self.current_state["second_open_stream_dest_node"] = self.second_open_stream_dest_node
                    self.current_state["pending_params"] = self.pending_params
                    return False, 0.0, True

            elif self.level == 3:
                # select amount for the select parameter of add solvent 
                if self._chosen_unit_name() == "add_solvent":
                    _, component, _, _ = self.pending_params["add_solvent"].values()
                    if component:
                        comp_map = self.env_config.add_solvent_comp_map[component]
                        self.selected_amount = comp_map[action_index]
                        self.pending_params["add_solvent"]["index_for_amount"] = action_index
                        self.pending_params["add_solvent"]["amount_value"] = self.selected_amount
                        self.history.append(action_index)

                        self.current_state["pending_params"] = self.pending_params
                        self.level = 0 
                        self.get_feasible_actions()
                        done, reward, worked = self._complete_action_place_and_simulate()
                        self.current_state["completed_design"] = done 
                        self.current_state["npv_norm"] = reward
                
                elif self._chosen_unit_name() == "mixer":
                    index_value, _, _ = self.pending_params["mixer"].values()
                    all_candidates = self._enumerate_open_streams_excluding(exclude = self.current_state["chosen_open_stream"])
                    src_node, _ = self.current_state["chosen_open_stream"]
                    true_cands = []
                    for i, out in all_candidates:
                        if i == index_value and i != src_node:
                            true_cands.append(out)
                    if not true_cands or action_index >= len(true_cands):
                        return False, 0.0, False  
                    out_value = true_cands[action_index]
                    self.pending_params["mixer"]["value_outlet"]= out_value
                    self.pending_params["mixer"]["tuple_value"] = (index_value, out_value)
                    self.history.append(action_index) 
                    self.current_state["pending_params"] = self.pending_params
                    self.second_open_stream = (index_value, out_value)
                    self.current_state["second_open_stream"] = self.second_open_stream

                    self.level = 0 
                    self.get_feasible_actions()
                    done, reward, worked = self._complete_action_place_and_simulate()
                    self.current_state["completed_design"] = done 
                    self.current_state["npv_norm"] = reward
 
                else:
                    if self._chosen_unit_name() not in ["add_solvent", "mixer"]:
                        raise RuntimeError("Only add_solvent or mixer related decisions allowed on Level 3")
                
                return done, reward, worked
            

        except Exception as e:
            # failed move (e.g., PEQ fail, convergence fail)
            print("take_action error:")
            print("  exception type:", type(e))
            print("  exception value:", e)
            traceback.print_exc()
            # reset to prevent level loops
            self._reset_action_state()
            return False, 0.0, False

        return False, 0.0, True

    # ------------- internals -------------#

    def _unit_available(self, unit_name: str, unit_idx: int) -> bool:
        if self.total_units_placed >= getattr(self.env_config, "max_total_units", 9999):
            return False

        cap_map = {
            "distillation_column": self.env_config.max_distillation_columns,
            "decanter": self.env_config.max_decanters,
            "split": self.env_config.max_split,
            "mixer": self.env_config.max_mixer,
            "recycle": self.env_config.max_recycle,
            "add_solvent": self.env_config.max_solvent,
        }
        if unit_name in cap_map and self.counts[unit_name] >= cap_map[unit_name]:
            return False

        # add_solvent: only allowed components (global indices)
        if unit_name == "add_solvent":
            start = self.env_config.add_solvent_start_index
            if start is None or unit_idx < start:
                return False
            comp_global_idx = unit_idx - start
            allowed = self.sim.feed_stream_information.get("possible_ind_add_comp", [])
            if comp_global_idx not in allowed:
                return False

        # mixer needs >=2 open streams
        if unit_name == "mixer":
            if len(self._enumerate_open_streams()) < 2:
                return False

        # recycle needs:
        #  - >=2 open streams
        #  - a chosen source stream (comes from level 0 first)
        #  - at least one eligible destination unit with single input
        if unit_name == "recycle":
            if len(self._enumerate_open_streams()) < 2:
                return False
            if self.current_state["chosen_open_stream"] is None:
                return False
            src_node, _ = self.current_state["chosen_open_stream"]
            if not self.sim.get_units_with_single_input(exclude=src_node):
                return False

        return True

    def _chosen_unit_name(self) -> Optional[str]:
        chosen_unit_index, _ = self.current_state["chosen_unit"]
        if chosen_unit_index is None:
            return None
        return self.env_config.units_map_indices_type[chosen_unit_index]

    def _enumerate_open_streams(self) -> List[Tuple[int, str]]:
        return self.sim.get_open_streams()

    def _enumerate_open_streams_excluding(self, exclude: Optional[Tuple[int, str]]) -> List[Tuple[int, str]]:
        all_ops = self.sim.get_open_streams()
        if exclude is None:
            return all_ops
        return [(n, l) for (n, l) in all_ops if not (n == exclude[0] and l == exclude[1])]


    def _assert_stream_is_open(self, stream: Tuple[int, str]) -> None:
        opens = set(self._enumerate_open_streams())
        if stream not in opens:
            raise RuntimeError(f"Chosen stream {stream} is no longer open.")

    def _complete_action_place_and_simulate(self) -> Tuple[bool, float, bool]:
        
        
        """
        Place the chosen unit (or recycle), wire edges, run simulate(), compute NPV.
        Resets to level 0 when done (unless terminated).
        
        """
        chosen_unit_index, unit_name = self.current_state["chosen_unit"]
        if self.current_state["chosen_open_stream"] is None or chosen_unit_index is None or unit_name is None:
            raise RuntimeError("Action incomplete (missing stream or unit).")

        # Ensure the source stream is still open
        self._assert_stream_is_open(self.current_state["chosen_open_stream"])
        src_node, src_label = self.current_state["chosen_open_stream"]

        # Build params per unit
        params: Dict[str, Any] = {}
        created_node_id: Optional[int] = None

        try:
            # Continuous param (if any)
            if unit_name == "distillation_column":
                _, cont_val = self.current_state["pending_params"]["distillation_column"]
                params["df"] = cont_val
            elif unit_name == "split":
                _, cont_val = self.current_state["pending_params"]["split"]
                params["split_ratio"] = cont_val

            elif unit_name == "add_solvent":
                index, component, index_for_amount, amount_value = self.current_state["pending_params"]["add_solvent"].values()
                cont_val = amount_value
                params = {
                    "index_new_component": index,
                    "solvent_amount": float(cont_val),
                }

            # Actually place
            if unit_name == "mixer":
                index, second_o_str_name = self.current_state["pending_params"]['mixer']
                if second_o_str_name is None:
                    raise RuntimeError("Mixer requires a second open stream.")
                n2, l2 = index, second_o_str_name
                created_node_id = self.sim.add_unit(
                    [(src_node, src_label), (n2, l2)],
                    "mixer",
                    params={},
                    num_outputs=1
                )

            elif unit_name == "recycle":
                recycle_dest = self.current_state["pending_params"]["recycle"]
                if recycle_dest is None:
                    raise RuntimeError("Recycle requires a destination unit to be chosen.")
                if recycle_dest == src_node:
                    raise ValueError("Cannot recycle a stream back into its own producing unit.")
                
                # Add recycle edge (transactional)
                self.sim.add_recycle(src_node, src_label, recycle_dest)
                created_node_id = None  # no new node

            elif unit_name == "add_solvent":
                created_node_id = self.sim.add_unit(
                    [(src_node, src_label)],
                    "add_solvent",
                    params=params,
                    num_outputs=1
                )

            elif unit_name == "distillation_column":
                created_node_id = self.sim.add_unit(
                    [(src_node, src_label)],
                    "distillation_column",
                    params=params,
                    num_outputs=2
                )

            elif unit_name == "decanter":
                created_node_id = self.sim.add_unit(
                    [(src_node, src_label)],
                    "decanter",
                    params={},
                    num_outputs=2
                )

            elif unit_name == "split":
                created_node_id = self.sim.add_unit(
                    [(src_node, src_label)],
                    "split",
                    params=params,
                    num_outputs=2
                )

            else:
                raise ValueError(f"Unknown unit type: {unit_name}")

            # simulate + NPV
            self.sim.simulate()

        except Exception as e:
            print ("Exception type:", type(e))
            print("Placement/simulation failed:", e)
            traceback.print_exc()

            # rollback if we placed a unit
            if created_node_id is not None:
                try:
                    self.sim.remove_node_and_restore_upstream_open(created_node_id)
                except Exception:
                    pass
            # rollback recycle edge if we placed one
            if unit_name == "recycle":
                try:
                    self.sim.remove_recycle_edge(src_node, src_label, self.recycle_dest_unit)
                except Exception:
                    pass

            # reset to avoid level loops
            #self._reset_action_state()
            return False, 0.0, False

        # update counts if worked and a *new* node was placed (recycle places no node)
        if unit_name == "recycle":
            self.counts["recycle"] += 1
        else:
            self.counts[unit_name] += 1
            self.total_units_placed += 1

        # reward = normalized NPV
        #reward = self.sim.current_net_present_value_normed or 0.0
        reward = self.sim.current_net_present_value or 0.0

        # reset for next turn
        #self._reset_action_state()

        # termination condition based on max units
        finished_design = self.total_units_placed >= getattr(self.env_config, "max_total_units", 9999)
        return finished_design, reward, True

    def _reset_action_state(self):
        self.current_state["current_level"] = None 
        self.current_state["open_streams"] = None 
        self.current_state["chosen_unit"] = None 
        self.current_state["chosen_open_stream"] = None 
        self.current_state["pending_params"] = None 
        self.current_state["npv_raw"] = None 
        self.current_state["npv_norm"] = None 
        self.current_state["completed_design"] = None 
        self.current_state["second_open_stream_dest_node"] = None
        self.current_state["second_open_stream"] = None
        self.current_state["recycle_dest_unit"] = None 
        self.current_state["recycler_mask"] = None
        self.current_state["mixer_mask"] = None
        self.current_state["current_action_mask"] = None


    def _eligible_recycle_destinations(self) -> List[int]:
        """
        Return the exact list of destination unit node_ids that are legal for the
        currently chosen open stream:
          - must be a “single-input” unit per simulator
          - must not be a feed
          - must not be the origin (producer) unit of the chosen stream
        """
        if self.current_state["chosen_open_stream"] is None:
            return []
        
        origin_node_id = self.current_state["chosen_open_stream"][0]
        dests = list(self.sim.get_units_with_single_input())  # simulator's base filter
        # exclude origin and feeds (can't recycle into feed)
        dests = [nid for nid in dests if nid != origin_node_id and nid not in getattr(self.sim, "feed_nodes", [])]
        return dests


    def compute_recycler_mask(self) -> torch.Tensor:

        """
        Make attention masks for recycler expert unit. 
        0 = not allowed, 1 = allowed 

        mask: torch.tensor of shape (num_nodes, num_nodes)

        """

        node_ids = list(self.sim.graph.nodes)
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        N = len(node_ids)
        mask = torch.zeros(N, N)

        # chosen source node id (graph-space)
        if self.current_state["chosen_open_stream"] == None: 
            return mask 
        else:
            source_node = self.current_state["chosen_open_stream"][0]
            src_idx = id_to_idx[source_node]

        # candidate destination node ids (graph-space)
        cand_node_ids = self._eligible_recycle_destinations()

        if len(cand_node_ids) == 0:
            return mask

        for dst_node_id in cand_node_ids:
            dst_idx = id_to_idx[dst_node_id]
            if dst_node_id != source_node:
                mask[src_idx, dst_idx] = 1 # avoid self connections and only allow valid dest candidates corresponding to the row of chosen unit''' 
        
        return mask

    
    def compute_mixer_mask(self) -> torch.Tensor:

        """
        Make attention masks for mixer expert unit. 
        0 = not allowed, 1 = allowed 

        """
        open_streams = self._enumerate_open_streams()
        open_nodes_id = [node_id for node_id, _ in open_streams]

        node_ids = list(self.sim.graph.nodes)
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        N = len(node_ids)

        mask = torch.zeros(N, N)

        src_node = self.current_state["chosen_open_stream"] 
        if src_node is None:
            return mask

        src_node_id, _ = src_node  
        src_idx = id_to_idx[src_node_id]

        for dst_node_id in open_nodes_id:
            dst_idx = id_to_idx[dst_node_id]
            if dst_node_id != src_node_id:
                mask[src_idx, dst_idx] = 1 # avoid self connections and only allow valid dest candidates corresponding to the row of chosen stream        
        return mask
    
    def masked_log_probs_for_current_action_level(self, logits: np.ndarray) -> np.ndarray:
        
        """
        Apply current_action_mask to logits and return normalized log-probs.
        """

        mask = self.current_action_mask.astype(bool)
        logits = logits.copy()
        logits[~mask] = -np.inf
        with np.errstate(divide="ignore", invalid="ignore"):
            log_probs = np.log(softmax(logits))

        return log_probs
    
    def is_terminable(self):
        return self.level == 0 and not self.finished_design
    

    @staticmethod
    def get_open_stream_mask_padded(flowsheets: List['FlowsheetDesign']):
        
        batch_open_streams = []

        for fs in flowsheets:
            open_streams = fs.current_state["open_streams"]
            batch_open_streams.append(open_streams)

        batch_open_node_indices = []
        open_stream_mask = []
        batch_open_outlet_indices = []

        for i, s in enumerate(batch_open_streams):
            if s is None:
                print(f"[ERROR] open_streams is None at batch index {i}")
                print(f"Flowsheet / state object: {flowsheets[i]}")
                raise RuntimeError("Found None open_streams")

        #batch_open_streams = [s or [] for s in batch_open_streams]
        max_open = max(len(s) for s in batch_open_streams)

        for streams, fs in zip(batch_open_streams, flowsheets):
            
            #canonical node ordering
            node_ids = list(fs.sim.graph.nodes)
            id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

            node_indices = []
            outlet_indices = []
            
            for node_id, outlet in streams:
                node_indices.append(id_to_idx[node_id])
                outlet_indices.append(fs.env_config.outlet_to_idx[outlet])

            pad = max_open - len(node_indices)
            node_indices += [0] * pad
            outlet_indices += [0] * pad
            mask = [True] * (len(streams)) + [False] * pad

            batch_open_node_indices.append(node_indices)
            batch_open_outlet_indices.append(outlet_indices)
            open_stream_mask.append(mask)

        batch_open_node_indices = torch.tensor(batch_open_node_indices, dtype=torch.long)
        open_stream_mask = torch.tensor(open_stream_mask, dtype=torch.bool)
        batch_open_outlet_indices = torch.tensor(batch_open_outlet_indices, dtype =torch.long)

        return batch_open_node_indices, open_stream_mask, batch_open_outlet_indices


    @staticmethod
    def get_embeddings_from_experts(flowsheets: List['FlowsheetDesign']):

        """
        Get embeddings in latent dim for each node and edge using predefined units and edge experts 

        """

        batch_latent_nodes_embeds, batch_latent_edges_embeds = [], []

        for fs in flowsheets: 
            node_ids = list(fs.sim.graph.nodes)
            id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
            N = len(node_ids)

            node_embeds = []

            for nid in node_ids:
                node_data = fs.sim.graph.nodes[nid]
                if node_data["unit_type"] == "feed":
                    emb = fs.open_stream_expert(node_data)
                else:
                    if node_data["unit_type"] in ["mixer", "decanter"]:
                        emb = fs.unit_experts[node_data["unit_type"]].embed(batch_size=1)
                    else:
                        emb = fs.unit_experts[node_data["unit_type"]].embed(node_data)
                
                node_embeds.append(emb)
            
            # edges 
            edge_embeds = torch.zeros(N + 1 , N + 1, fs.gen_config.latent_dim, device=fs.gen_config.training_device) # +1 for handling edges for virtual node 
            
            # fill everything with "no edge"
            no_edge_emb = fs.edge_expert(
                edge_exists=False,
                is_recycle=False, 
                carries_flow = False, 
            )

            edge_embeds[:] = no_edge_emb

            # now overwrite where real edges exist
            for u, v, edge_data in fs.sim.graph.edges(data=True):
                ui = id_to_idx[u]
                vi = id_to_idx[v]
                if "unit_type" in edge_data:
                    edge_embeds[ui, vi] = fs.edge_expert(edge=edge_data, edge_exists=True, is_recycle = edge_data.get("is_recycle", False), carries_flow= True)
                else:
                    edge_embeds[ui, vi] = fs.edge_expert(edge=edge_data, edge_exists=True, is_recycle = edge_data.get("is_recycle"), carries_flow= False)

            # Extract open stream related information from the fs
            batch_open_node_indices, open_stream_mask, batch_open_outlet_indices = FlowsheetDesign.get_open_stream_mask_padded(flowsheets)
            batch_latent_nodes_embeds.append(node_embeds)
            batch_latent_edges_embeds.append(edge_embeds)

        return dict(batch_latent_nodes_embeds = batch_latent_nodes_embeds, 
                batch_latent_edges_embeds = batch_latent_edges_embeds, 
                batch_open_node_indices = batch_open_node_indices, 
                batch_open_outlet_indices = batch_open_outlet_indices, 
                open_stream_mask = open_stream_mask)
    
    # ---- Implementation of abstract methods from `BaseTrajectory`
    def transition_fn(self, action: int) -> Tuple['BaseTrajectory', bool]:
        copied_fs= copy.deepcopy(self)
        copied_fs.take_action(action)
        return copied_fs, copied_fs.finished_design
    
    def to_max_evaluation_fn(self) -> float:
        if self.current_state["npv_raw"] is None:
            raise ValueError("Objective is `None`. Check if Flowsheet Simulator really works")
        return self.current_state["npv_raw"]
    
    def make_mask_list_as_stacked_tensors(masking_list, whether_mixer, flowsheets):

        if all(m is None for m in masking_list):
            return None 
    
        max_len = max(m.shape[0] for m in masking_list if m is not None)
        masks = []

        for m, fs in zip(masking_list, flowsheets):
            if m is None:
                # add a new padded mask 
                padded = torch.zeros(
                    (max_len, max_len))
            else:
                n = m.shape[0]
                if n != max_len:
                    padded = torch.zeros((max_len, max_len), device=m.device)
                    padded[:n, :n] = m
                else:
                    padded = m
            
            if whether_mixer:
                fs.padded_mixer_mask = padded
            else:
                fs.padded_recycler_mask = padded
            
            masks.append(padded)

        return torch.stack(masks, dim = 0)
    
    @staticmethod
    def log_probability_fn(trajectories: List['FlowsheetDesign'], network: nn.Module, device: torch.device, config) -> List[np.array]:
        
        """
        Given a list of trajectories and a policy network,
        returns a list of numpy arrays, each having length num_actions, where each numpy array is a log-probability
        distribution over the next action level.

        Parameters:
            trajectories [List[BaseTrajectory]]
            network [torch.nn.Module]: Policy network
        Returns:
            List of numpy arrays, where i-th entry corresponds to the log-probabilities for i-th trajectory.

        """
        log_probs_to_return: List[np.array] = []
        device = torch.device("cpu") if device is None else device
        network.eval()
        with torch.no_grad():
            with torch.amp.autocast(device_type=config.training_device):
                batch = FlowsheetDesign.list_to_batch(flowsheets=trajectories, device=network.device)
                lvl_0_logits, unit_predictions, padded_open_stream_masks, valid_nodes = network(batch)
                for i, fs in enumerate(trajectories):
                    # get logits for this sequence and corresponding level
                    if fs.level == 0:
                        logit = lvl_0_logits[i] # (N+1,)
                        if len(fs.current_action_mask) != logit.shape[0]:
                            terminate_logits = logit[0]
                            open_stream_logits= logit[1:]
                            open_stream_valid_logits = open_stream_logits[padded_open_stream_masks[i]]
                            logits = torch.cat([terminate_logits.unsqueeze(0), open_stream_valid_logits],dim=0)
                        else:
                            logits = logit
                        logits = np.array(logits.float())
                    
                    if fs.level == 1:
                        node_id, _ = fs.current_state["chosen_open_stream"]
                        # collect logits for units corresponding to ONLY this open stream 
                        logits = []
                        for _, unit_name in enumerate(fs.env_config.units_map_indices_type):
                            logit = unit_predictions[unit_name]["picked_logit"][i, node_id, 0] # scalar logit per unit for ith batch, at node node_id and removing any singleton dimension
                            logits.append(logit.float())
                        logits = np.array(logits) # (length of all possible units, _)
                    
                    if fs.level == 2: # extract parameter predictions 
                        unit_id, unit_name = fs.current_state["chosen_unit"]
                        node_id, _ = fs.current_state["chosen_open_stream"]
                        logits = []
                        if unit_name == "distillation_column":
                            logits = unit_predictions[unit_name]["distillate_fraction_categorical"][i, node_id, :]
                        if unit_name == "mixer":
                            target_scores = unit_predictions[unit_name]["target_scores"][i, node_id, :]
                            #logits = target_scores[0:len(fs.current_action_mask)]
                            logits = target_scores[valid_nodes[i][1:]]
                        if unit_name == "recycle":
                            target_scores = unit_predictions[unit_name]["target_scores"][i, node_id, :]
                            logits = target_scores[valid_nodes[i][1:]]
                        if unit_name == "split":
                            logits = unit_predictions[unit_name]["split_ratio_categorical"][i, node_id, :]
                        if unit_name == "add_solvent":
                            logits = unit_predictions[unit_name]["component_logit"][i, node_id, :] 
                        
                        logits = np.array(logits.float())

                    if fs.level == 3:
                        logits = []
                        unit_id, unit_name = fs.current_state["chosen_unit"]
                        node_id, _ = fs.current_state["chosen_open_stream"]

                        if unit_name == "add_solvent":
                            index_comp, comp_name, _, _ = fs.current_state["pending_params"]["add_solvent"].values()
                            logits = unit_predictions[unit_name]["component_amount"][i, node_id, index_comp, :]
                        elif unit_name == "mixer":
                            dest_node, out_value = fs.current_state["second_open_stream"]
                            outlet_logits = unit_predictions[unit_name]["destinate_node_outlets"][i, dest_node, 0:len(fs.current_action_mask)]
                            logits = outlet_logits

                        logits = np.array(logits.float())
                    
                    log_probs_to_return.append(fs.masked_log_probs_for_current_action_level(logits))
        return log_probs_to_return
    
    @staticmethod
    def batch_to_device(batch: dict, device: torch.device):
        """
        Takes batch as returned from `list_to_batch` and moves it onto the given device.
        """
        return {k: v.to(device) for k, v in batch.items()}
    
    @staticmethod
    def design_flowsheets(random_instance: Dict[str, Any], gen_config, env_config) -> List['FlowsheetDesign']:

        """
        Returns list of flowsheet designs based on a starting problem instance 

        """
        instance_list = []
        flowsheet_traj = FlowsheetDesign(random_instance, gen_config, env_config)
        instance_list.append(flowsheet_traj)
        return instance_list
    
    @staticmethod
    def list_to_batch(flowsheets: List['FlowsheetDesign'], include_feasibility_masks: bool = False, device: torch.device = None) -> dict:
        
        """
        Given a list of flowsheet graphs, prepares a batch that can be passed through the network.

        The batch is given as a dictionary with the following keys and values:
        * "level_idx": tensor containing information about the current level that the agent is on 
        * "batch_latent_nodes": stacked tensors of node embeddings, as processed from unit/open_stream experts
        * "batch_latent_edge": stacked tensors of edge embeddings, as processed from edge experts
        * "recycler_masks": stacked tensors of padded recycler_masks, which will be passed to recycle unit expert
        * "mixer_masks": stacked tensors of padded mixer_masks, which will be passed to mixer unit expert
        * "node_valid_masks": stacked tensors of padded node masks, denoting whether something is a real node 
        * "edge_valid_masks": stacked tensors of padded edge masks, denoting whether something is a real edge 
        * "batch_open_node_indices": tensors denoting the indices of nodes that provide information about current open streams. Will be useful to extract 
                                    embeddings of the corresponding nodes 
        *  "batch_open_outlet_indices": tensors denoting the indices of open streams. can be either 0 for out0 or 1 for out1
        * "open_stream_mask": a boolen tensor denoting whether something is a real open stream

        if `include_feasibility_masks` is set to True, we also return
        """

        assert len(flowsheets) > 0, "Empty batch of flowsheets"

        # Calculate latent dimension from experts  
        embeddings_dict = FlowsheetDesign.get_embeddings_from_experts(flowsheets = flowsheets)
        batch_nodes, batch_edges = [], []
        
        # stack nodes within flowsheets 
        for ns, es in zip(embeddings_dict["batch_latent_nodes_embeds"], embeddings_dict["batch_latent_edges_embeds"]):
            ns = [n.squeeze(0) for n in ns] # make dimension (512, )
            fs_nodes = torch.stack(ns, dim=0)   # (N, d)
            fs_edges = es
            batch_nodes.append(fs_nodes)
            batch_edges.append(fs_edges)

        # mask for node validity
        B = len(batch_nodes)
        N_max = max(nodes.shape[0] for nodes in batch_nodes)
        device = batch_nodes[0].device
        node_valid_mask = torch.zeros(B, N_max + 1, dtype=torch.bool, device=device)
        
        for b, nodes in enumerate(batch_nodes):
            n = nodes.shape[0]
            node_valid_mask[b, 0] = True          # virtual node (always valid)
            node_valid_mask[b, 1 : 1 + n] = True  # real nodes
        
        batch_nodes_padded = pad_sequence(batch_nodes, batch_first=True, padding_value=0.0)

        # padding for edges 
        B = len(batch_edges)
        N_max = max(e.shape[0] for e in batch_edges)
        d_edge = batch_edges[0].shape[-1]
        batch_edges_padded = torch.zeros(B, N_max, N_max, d_edge, device=device)

        for b, edges in enumerate(batch_edges):
            n = edges.shape[0]
            batch_edges_padded[b, :n, :n] = edges

        edge_valid_mask = (node_valid_mask[:, :, None] & node_valid_mask[:, None, :])

        # Collect recycle and mixer masks 
        for i, fs in enumerate(flowsheets):
            fs.current_state["mixer_mask"] = fs.compute_mixer_mask()
            fs.current_state["recycler_mask"] = fs.compute_recycler_mask()
            fs.valid_nodes = node_valid_mask[i]

        recycler_masks = [fs.current_state["recycler_mask"] for fs in flowsheets]     
        mixer_masks = [fs.current_state["mixer_mask"] for fs in flowsheets]

        recycler_masks = FlowsheetDesign.make_mask_list_as_stacked_tensors(recycler_masks, whether_mixer = False, flowsheets= flowsheets)
        mixer_masks = FlowsheetDesign.make_mask_list_as_stacked_tensors(mixer_masks, whether_mixer = True, flowsheets= flowsheets)

        return_dict = dict(
            state_information= [fs.current_state for fs in flowsheets],  # (B,)
            batch_latent_nodes = batch_nodes_padded,                  # (B, num_nodes))
            batch_latent_edges = batch_edges_padded,
            recycler_masks = recycler_masks,        
            mixer_masks = mixer_masks, 
            node_valid_mask = node_valid_mask, 
            edge_valid_mask = edge_valid_mask,
            batch_open_node_indices = embeddings_dict["batch_open_node_indices"],
            batch_open_outlet_indices = embeddings_dict["batch_open_outlet_indices"],
            open_stream_mask = embeddings_dict["open_stream_mask"],
        )

        '''if include_feasibility_masks:
            # Build per-level feasibility masks, padded across the batch to each level's max action count.
            feasibility_mask_per_level = []

            num_actions_per_level_and_seq = [
                [len(seq.residues) for seq in sequences],  # lvl 0 
                [len(seq.vocabulary_residue_names) + 1 for seq in sequences],  # lvl 1
            ]

            for lvl, num_actions_per_seq in enumerate(num_actions_per_level_and_seq):
                max_num_actions = max(num_actions_per_seq)
                feasibility_mask_per_level.append(
                torch.from_numpy(
                    np.stack([
                        np.pad(
                            seq.current_action_mask,
                            (0, max_num_actions - num_actions_per_seq[i]),
                            mode='constant', constant_values=1
                        ) if seq.current_action_level == lvl 
                        else np.zeros(max_num_actions, dtype=bool)
                        for i, seq in enumerate(sequences)
                    ])
                ).bool().to(device)
            )
        
            # Add to return_dict
            return_dict["feasibility_mask_level_zero"] = feasibility_mask_per_level[0] 
            return_dict["feasibility_mask_level_one"] = feasibility_mask_per_level[1]  '''

        return return_dict
