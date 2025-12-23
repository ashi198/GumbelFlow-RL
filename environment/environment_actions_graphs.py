# environment_actions_graph.py
# how to implement history list here 

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from config import GeneralConfig, EnvConfig
from core.abstract import BaseTrajectory
from core.utils import softmax
import numpy as np
from environment.flowsheet_simulation_graph import FlowsheetSimulationGraph
import copy, torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from experts.experts import DistillationColumn, Decanter, Mixer, Split, Recycler, AddSolvent, EdgeFlowExpert, OpenStreamExpert



class FlowsheetDesign:

    """
    Graph-based action environment:

      Level 0: choose OPEN stream (or index 0 = TERMINATE)
      Level 1: choose UNIT type and its parameters 

    After an action completes successfully:
      - placement/wiring happens (add_unit / add_recycle)
      - simulate() runs
      - NPV is recomputed
      - state resets to level 0

    On any failure:
      - we roll back partial changes (including recycle edges)
      - reset to level 0 (no loops)
    """

    def __init__(self, random_instance: Dict[str, Any]):
        self.gen_config = GeneralConfig
        self.env_config = EnvConfig 
        self.sim = FlowsheetSimulationGraph(random_instance, self.env_config)
        self.num_units = len(self.env_config.unit_types)

        # add feed nodes
        self.sim.feed_nodes = []
        for feed in random_instance["list_feed_streams"]:
            self.sim.feed_nodes.append(self.sim.add_feed(feed))

        # List of mappings for params for distillation, decanter, add_solvent, recycler 
        self.DF_distillation_map = np.linspace(0.01, 0.99, 100)
        self.split_ratio_map = np.linspace(0.01, 0.99, 100)
        self.Tc_comp_map = np.linspace([0.01, 10], 100)
        self.Pc_comp_map = np.linspace([0.01, 10], 100)
        self.omega_comp_map = np.linspace([0.01, 10], 100)
        self.a_comp_map = np.linspace([0.01, 10], 100)

        self.add_solvent_comp_map = {
            0: ("Tc_comp", self.Tc_comp_map),
            1: ("Pc_comp", self.Pc_comp_map), 
            2: ("omega_comp", self.omega_comp_map), 
            3: ("a_comp", self.a_comp_map), 
                } 

        # params variables 
        self.DF_distillation: float = None 
        self.split_ratio: float = None 
        self.Tc_comp: float = None 
        self.Pc_comp: float = None 
        self.omega_comp: float = None 
        self.a_comp: float = None 

        # current action state
        self.level = 0 
        self.chosen_open_stream: Optional[Tuple[int, str]] = None    # (node_id, label)
        self.chosen_unit: Optional[Tuple[int, str]]                  # index in units_map_indices_type
        self.pending_params: Optional[Dict] = {}  # params for the chosen unit  
        self.second_open_stream: Optional[Tuple[int, str]] = None     # for mixer
        self.recycle_dest_unit: Optional[int] = None                  # for recycle

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
            "add_solvent": AddSolvent(self.gen_config),    
            "open_stream": OpenStreamExpert(self.gen_config)     
            }
        
        self.edge_expert = EdgeFlowExpert()
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

        self.objective = self.current_state["npv_norm"]


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
            "second_open_stream": self.second_open_stream,
            "recycle_dest_unit": self.recycle_dest_unit

        }
        return state

    def get_feasible_actions(self) -> np.ndarray:

        """
        Return a 0/1 vector mask feasible actions for the current level:

        Level 0: indices enumerate all open streams, plus 1 extra for "terminate"
        Level 1: indices enumerate unit choices (0...num_units-1) 
        Level 2: select parameters values for the following units (distillation column: DF value, split: split_ratio, mixer: 2nd open stream, add_solvent: parameter
        recycler: destination stream)
        Level 3: select amount for selected parameter for add_solvent  

        0: the action is masked. 1 = action is allowed

        """
        if self.level == 0:
            open_streams = self.current_state["open_streams"]
            mask = np.zeros(len(open_streams) + 1, dtype=int) #check if this is valid 
            
            # masking condition for lvl 0 (terminate)
            if self.finished_design == True: 
                mask[-1] = 1  
                mask[:len(open_streams)] = 0
            else:
                mask[-1] = 0

            # enable all available stream slots (everything can be available to be changed)
            mask[:len(open_streams)] = 1
            self.current_action_mask = mask

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
                        unit_params_mask[idx] = 1 
                if unit_name == "recycle" and avail_unit:
                    # need a chosen stream (comes from level 0) and at least one eligible dest
                    if self.current_state["chosen_open_stream"] is None or len(self._eligible_recycle_destinations()) == 0:
                        continue
                    else:
                        unit_params_mask[idx] = 1

            self.current_action_mask = unit_params_mask
            return unit_params_mask

        elif self.level == 2:
            open_streams = self.current_state["open_streams"] + 1
            _, chosen_unit_name = self.current_state["chosen_unit"]
            if chosen_unit_name in ["distillation", "split"]:
                params_mask = np.ones(100, dtype=int) # what constraints can be present here?
            elif chosen_unit_name == "add_solvent":
                params_mask = np.ones(4, dtype= int) # what constraints can be present here?
            elif chosen_unit_name == "recycle": # should recycle be better open_streams??
                params_mask = np.zeros(self.num_units, dtype=int)
                recycle_dests = self._eligible_recycle_destinations()
                for unit_id in recycle_dests:
                    params_mask[unit_id] = 1
            elif chosen_unit_name == "mixer":
                params_mask = np.zeros((len(open_streams), len(open_streams)), dtype=int) # check this later 
                candidates = self._enumerate_open_streams_excluding(self.chosen_open_stream)
                chosen_open_stream_index, _ = self.chosen_open_stream
                for j in candidates:
                    params_mask[chosen_open_stream_index][j] = 1 
            self.current_action_mask = params_mask
            return params_mask

        elif self.level == 3 and self.current_state["chosen_unit_name"] == "add_solvent":
            params_mask =  np.ones(100, dtype=int)
            self.current_action_mask = params_mask #now decide components for parameters

            return params_mask
        
        return np.array([], dtype=int)

    def take_action(self, action_index: int) -> Tuple[bool, float, bool]:

        """
        Returns:
          finished_design (bool) = true if termination chosen or max units reached
          reward (float) = current (normalized) NPV after completing an action
          move_worked (bool) = false if placement failed (e.g., PEQ failure / convergence fail)

        """

        assert not self.finished_design, "Taking action on an already terminated design!"
        assert self.current_action_mask[action_index] == 0, \
            f"Trying to take action {action_index} on level {self.level}, but it is set to infeasible"
        if action_index >= len(self.current_action_mask):
            raise ValueError(f"Invalid action {action_index}, mask size {len(self.current_action_mask)}") 
        
        try:
            if self.level == 0:
                open_streams = self._enumerate_open_streams()      
                if action_index == 0:  # Check this later 
                    self.finished_design = True 
                    self.history.extend([action_index] + [self.SKIP_ACTION] * 3) # because terminate do not have lvl1, lvl 2, lvl 3 decisions, we assign "skip_action" as 901
                    return True, (self.sim.current_net_present_value_normed or 0.0), True
                
                # if not terminate index, then open_stream selected 
                stream_index, stream_name = open_streams
                self.chosen_open_stream = stream_index[action_index - 1], stream_name[action_index - 1]
                self.level = 1
                self.history.append(action_index)
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
                    self.pending_params["distillation_column"] = (int, int) # tuple of index, value
                    self.level = 2
                    self.chosen_unit = (unit_idx, unit_name)
                    self.history.append(action_index)
                    return False, 0.0, True
                
                if unit_name == "split":
                    # add a new category for DF ratio
                    self.pending_params["split"] = (int, int) # tuple of index, value
                    self.level = 2
                    self.chosen_unit = (unit_idx, unit_name)
                    self.history.append(action_index)
                    return False, 0.0, True

                if unit_name == "add_solvent":
                    self.pending_params["add_solvent"] = {}
                    self.level = 2
                    self.chosen_unit = (unit_idx, unit_name)
                    self.history.append(action_index)
                    return False, 0.0, True

                # recycle always goes to level 2 (destination selection)
                if unit_name == "recycle":
                    self.pending_params["recycle"] = (int) # index of choosen destination
                    self.level = 2
                    self.chosen_unit = (unit_idx, unit_name)
                    self.history.append(action_index)
                    return False, 0.0, True
                
                if unit_name == "mixer":
                    self.pending_params['mixer'] = (int, str) # tuple of index, value
                    self.level = 2
                    self.chosen_unit = (unit_idx, unit_name)
                    self.history.append(action_index)
                    return False, 0.0, True 

                if unit_name == "decantor":
                    # immediate place (no continuous param, no second stream)
                    self.chosen_unit = (unit_idx, unit_name)
                    self.history.extend([action_index] + [self.SKIP_ACTION] * 2) 
                    done, reward, worked = self._complete_action_place_and_simulate()
                    return done, reward, worked

            elif self.level == 2:
                # for distillation, decide on which DF ratio value to choose 
                if self._chosen_unit_name() == "distillation_column":
                    if action_index > len (self.DF_distillation_map):
                        raise ValueError("Distillation fraction value selected more than the permissible limit.")
                    else:
                        self.DF_distillation = self.DF_distillation_map[action_index]
                        self.pending_params["distillation_column"] = (action_index, self.DF_distillation)
                        self.history.extend([action_index] + [self.SKIP_ACTION]) 
                        return self._complete_action_place_and_simulate()

                # for selecting split ratio 
                elif self._chosen_unit_name() == "split":
                    if action_index > len(self.split_ratio_map):
                        raise ValueError("Split ratio value selected more than the permissible limit.")
                    else:
                        self.split_ratio = self.split_ratio_map[action_index]
                        self.pending_params["split"] = (action_index, self.split_ratio)
                        self.history.extend([action_index] + [self.SKIP_ACTION]) 
                        return self._complete_action_place_and_simulate()
                    
                elif self._chosen_unit_name() == "add_solvent":
                    if action_index > 4:
                        raise ValueError("Invalid selection for thermodynamic components")
                    else:
                        self.chosen_add_solvent_comp = self.add_solvent_comp_map[action_index]
                        self.pending_params["add_solvent"] = {
                            "index_for_comp": action_index,
                            "name_comp": self.chosen_add_solvent_comp,
                            "index_for_amount": None,
                            "amount_value": None,
                        }
                        self.history.append(action_index)
                        self.level = 3 
                        
                # choose destination stream for recycle
                elif self._chosen_unit_name() == "recycle":
                    dests = self._eligible_recycle_destinations()
                    if action_index > len(dests):
                        raise ValueError("Invalid selection for destination stream for recycle ")
                    else:
                        # select destination and simulate 
                        self.recycle_dest_unit = dests[action_index]
                        self.pending_params["recycle"] = (self.recycle_dest_unit)
                        self.history.extend([action_index] + [self.SKIP_ACTION]) 
                        return self._complete_action_place_and_simulate()

                # mixer: choose second stream
                elif self._chosen_unit_name() == "mixer":
                    candidates = self._enumerate_open_streams_excluding(self.chosen_open_stream)
                    self.second_open_stream = candidates[action_index]
                    self.pending_params["mixer"] = self.second_open_stream
                    self.history.extend([action_index] + [self.SKIP_ACTION]) 
                    return self._complete_action_place_and_simulate()

            elif self.level == 3:
                # select amount for the select parameter of add solvent 
                if self._chosen_unit_name() != "add_solvent":
                    raise RuntimeError("Only add_solvent related decision allowed on Level 3")
                else:
                    index, component, _, _ = self.pending_params["add_solvent"]
                    if component:
                        _, comp_map = self.add_solvent_comp_map[index]
                        self.selected_amount = comp_map[action_index]
                self.pending_params["add_solvent"]["index_for_comp"] = action_index
                self.pending_params["add_solvent"]["amount_value"] = self.selected_amount
                self.history.append(action_index)
                return self._complete_action_place_and_simulate()
            
            self.get_feasible_actions()

        except Exception as e:
            # failed move (e.g., PEQ fail, convergence fail)
            print("take_action error:", e)
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
            if self.chosen_open_stream is None:
                return False
            src_node, _ = self.chosen_open_stream
            if not self.sim.get_units_with_single_input(exclude=src_node):
                return False

        return True

    def _chosen_unit_name(self) -> Optional[str]:
        if self.chosen_unit_index is None:
            return None
        return self.env_config.units_map_indices_type[self.chosen_unit_index]

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
        unit_name = self._chosen_unit_name()
        if self.chosen_open_stream is None or self.chosen_unit_index is None or unit_name is None:
            raise RuntimeError("Action incomplete (missing stream or unit).")

        # Ensure the source stream is still open
        self._assert_stream_is_open(self.chosen_open_stream)
        src_node, src_label = self.chosen_open_stream

        # Build params per unit
        params: Dict[str, Any] = {}
        created_node_id: Optional[int] = None

        try:
            # Continuous param (if any)
            if unit_name == "distillation_column":
                _, cont_val = self.pending_params["distillation_column"]
                params["df"] = cont_val
            elif unit_name == "split":
                _, cont_val = self.pending_params["split"]
                params["split_ratio"] = cont_val

            elif unit_name == "add_solvent":
                index, component, index_for_amount, amount_value = self.pending_params["add_solvent"]
                cont_val = amount_value
                params = {
                    "index_new_component": index,
                    "solvent_amount": float(cont_val),
                }

            # Actually place
            if unit_name == "mixer":
                index, second_o_str_name = self.pending_params['mixer']
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
                recycle_dest = self.pending_params["recycle"]
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
            print("Placement/simulation failed:", e)
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
            self._reset_action_state()
            return False, 0.0, False

        # update counts if worked and a *new* node was placed (recycle places no node)
        if unit_name == "recycle":
            self.counts["recycle"] += 1
        else:
            self.counts[unit_name] += 1
            self.total_units_placed += 1

        # reward = normalized NPV
        reward = self.sim.current_net_present_value_normed or 0.0

        # reset for next turn
        self._reset_action_state()

        # termination condition based on max units
        finished_design = self.total_units_placed >= getattr(self.env_config, "max_total_units", 9999)
        return finished_design, reward, True

    def _reset_action_state(self):
        self.level = None 
        self.chosen_open_stream = None
        self.chosen_open_stream_index = None 
        self.chosen_unit_index = None
        self.chosen_unit_name = None 
        self.second_open_stream = None
        self.recycle_dest_unit = None
        self.pending_params = {}
        self.second_open_stream = None 
        self.recycle_dest_unit = None 

    def _eligible_recycle_destinations(self) -> List[int]:
        """
        Return the exact list of destination unit node_ids that are legal for the
        currently chosen open stream:
          - must be a “single-input” unit per simulator
          - must not be a feed
          - must not be the origin (producer) unit of the chosen stream
        """
        if self.chosen_open_stream is None:
            return []
        origin_node_id = self.chosen_open_stream[0]
        dests = list(self.sim.get_units_with_single_input())  # simulator's base filter
        # exclude origin and feeds (can't recycle into feed)
        dests = [nid for nid in dests if nid != origin_node_id and nid not in getattr(self.sim, "feed_nodes", [])]
        return dests
    
    def masked_log_probs_for_current_action_level(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply current_action_mask to logits and return normalized log-probs.
        """
        mask = self.current_action_mask
        logits = logits.copy()
        logits[mask] = -np.inf
        with np.errstate(divide="ignore", invalid="ignore"):
            log_probs = np.log(softmax(logits))

        return log_probs
    
    def get_embeddings_from_experts(self, flowsheets: List['FlowsheetDesign']):

        """
        Get embeddings in latent dim for each node and edge using predefined units and edge experts 

        """
        batch_latent_nodes_embeds, batch_latent_edges_embeds = [], []
        for fs in flowsheets:
            latent_node_embed = [self.unit_experts[node.type].embed(node) for node in fs]
            latent_edge_embed = [self.edge_expert(edge) for edge in fs]
            batch_latent_nodes_embeds.append(latent_node_embed)
            batch_latent_edges_embeds.append(latent_edge_embed)

        return batch_latent_nodes_embeds, batch_latent_edges_embeds
    
    # ---- Implementation of abstract methods from `BaseTrajectory`
    def transition_fn(self, action: int) -> Tuple['BaseTrajectory', bool]:
        copied_fs= copy.deepcopy(self)
        copied_fs.take_action(action)
        return copied_fs, copied_fs.finished_design
    
    def to_max_evaluation_fn(self) -> float:
        if self.objective is None:
            raise ValueError("Objective is `None`. Check if Flowsheet Simulator really works")
        return self.objective
    
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
                batch = FlowsheetDesign.list_to_batch(sequences=trajectories, device=network.device)
                batch_logits_per_level = list(network(batch))
                for lvl in range(4):
                    batch_logits_per_level[lvl] = batch_logits_per_level[lvl].to(torch.float32).cpu().numpy()
                for i, fs in enumerate(trajectories):
                    # get logits for this sequence and corresponding level
                    if fs.level == 0:
                        logits = batch_logits_per_level[fs.level][i]
                    if fs.level == 1:
                        logits = batch_logits_per_level[fs.level][i]
                    if fs.level == 2: 
                        logits = batch_logits_per_level[fs.level][i] # check this
                    if fs.level == 3:
                        logits = batch_logits_per_level[fs.level][i] # check this 
                    log_probs_to_return.append(fs.masked_log_probs_for_current_action_level(logits))
        return log_probs_to_return
    
    @staticmethod
    def batch_to_device(batch: dict, device: torch.device):
        """
        Takes batch as returned from `list_to_batch` and moves it onto the given device.
        """
        return {k: v.to(device) for k, v in batch.items()}
    
    @staticmethod
    def design_flowsheets(random_instance: Dict[str, Any]) -> List['FlowsheetDesign']:

        """
        Returns list of flowsheet designs based on a starting problem instance 

        """
        instance_list = []
        flowsheet_traj = FlowsheetDesign(random_instance)
        instance_list.append(flowsheet_traj)
        return instance_list
    
    @staticmethod
    def list_to_batch(flowsheets: List['FlowsheetDesign'], include_feasibility_masks: bool = False, device: torch.device = None) -> dict:
        """
        Given a list of sequence designs, prepares a batch that can be passed through the network.

        The batch is given as a dictionary with the following keys and values:
        * "level_idx": This is "0" for level 0, "1" for level 1. Used to mark the virtual residue to 
        inform the network which decision to make.  

        * "tokens_np": either token ids or direct embeddings from ESM model

        * "selected_position": a list of selected position 

        if `include_feasibility_masks` is set to True, we also return
        """

        assert len(flowsheets) > 0, "Empty batch of flowsheets"
        pad_idx = 902 #to add padding to make all sequences of equal length

        # Calculate latent dimension from experts  
        batch_latent_nodes_embeds, batch_latent_edges_embeds = FlowsheetDesign.get_embeddings_from_experts(flowsheets)
        batch_latent_nodes_embeds = pad_sequence(batch_latent_nodes_embeds, batch_first = True, padding_value = pad_idx)
        batch_latent_edges_embeds = pad_sequence(batch_latent_edges_embeds, batch_first = True, padding_value = pad_idx)


        return_dict = dict(
            state_information= [fs.current_state for fs in flowsheets],  # (B,)
            batch_latent_nodes = torch.from_numpy(batch_latent_nodes_embeds).to(device),                  # (B, num_residues))
            batch_latent_edges = torch.from_numpy(batch_latent_edges_embeds).to(device),
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
