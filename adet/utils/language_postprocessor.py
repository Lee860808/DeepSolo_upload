# adet/utils/language_postprocessor.py

import torch
import torch.nn.functional as F
import numpy as np
import pickle
# import editdistance # Not strictly needed for these decoding methods
import heapq
import math
import os
import logging
import pdb # Keep for debugging if needed

try:
    import kenlm
except ImportError:
    print("*"*50)
    print("Please install kenlm python wrapper: ...") # Keep your instructions
    print("*"*50)
    raise

class BeamHypothesis:
    # ... (BeamHypothesis class as before) ...
    def __init__(self, sequence_indices, text, score, last_char_index, lm_state=None):
        self.sequence_indices = sequence_indices
        self.text = text
        self.score = score
        self.last_char_index = last_char_index
        self.lm_state = lm_state

    def __lt__(self, other):
        return self.score > other.score

    def __repr__(self):
        return f"Hyp(text='{self.text}', score={self.score:.4f})"

class LanguagePostProcessor:
    def __init__(self, cfg, lm_path=None, lm_weight=0.1, beam_width=5):
        self.logger = logging.getLogger("adet.LanguagePostProcessor")
        self.cfg = cfg
        self.voc_size_config = cfg.MODEL.TRANSFORMER.VOC_SIZE
        self.use_customer_dict = cfg.MODEL.TRANSFORMER.CUSTOM_DICT
        self._load_char_labels() # Sets self.CTLABELS and self.voc_size = len(self.CTLABELS)
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.blank_idx = self.voc_size_config # CTC Blank is at index defined by original config VOC_SIZE

        self.lm_path = lm_path
        self.lm_weight = lm_weight
        self.beam_width = beam_width # This will now control which decoding is used
        self.language_model = None
        self.lm_order = 0

        self.logger.info("-" * 50)
        self.logger.info(f"LanguagePostProcessor Init:")
        self.logger.info(f"  Configured LM Path: {self.lm_path}")
        self.logger.info(f"  LM Weight: {self.lm_weight}")
        self.logger.info(f"  Beam Width: {self.beam_width} (type: {type(self.beam_width)})")

        if self.lm_path:
            if not os.path.exists(self.lm_path):
                self.logger.warning(f"  WARNING: LM path '{self.lm_path}' not found. LM features disabled.")
                self.language_model = None
            else:
                try:
                    self.logger.info(f"  Attempting to load KenLM model from: {self.lm_path}")
                    self.language_model = kenlm.Model(self.lm_path)
                    self.lm_order = self.language_model.order
                    self.logger.info(f"  KenLM model loaded. Order: {self.lm_order}, Type: {type(self.language_model)}")
                except Exception as e:
                    self.logger.error(f"  ERROR loading KenLM model: {e}", exc_info=True)
                    self.language_model = None
        else:
            self.logger.info("  No LM path provided. LM features disabled.")

        if self.language_model is None:
             self.logger.warning("  WARNING: No KenLM model available. Post-processing will use pure greedy decoding if beam_width=1, or beam search without LM if beam_width>1.")
        self.logger.info(f"  Final self.language_model is None: {self.language_model is None}")
        self.logger.info(f"  Final self.lm_order: {self.lm_order}")
        self.logger.info("-" * 50)

    def _load_char_labels(self):
        # ... (Your _load_char_labels method as previously corrected) ...
        if self.voc_size_config == 37 and not self.use_customer_dict:
            self.CTLABELS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                             't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        elif self.voc_size_config == 96 and not self.use_customer_dict:
             self.CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']
        elif self.use_customer_dict:
             with open(self.use_customer_dict, 'rb') as fp:
                 self.CTLABELS = pickle.load(fp)
             if len(self.CTLABELS) != int(self.voc_size_config):
                 raise ValueError(f"Custom dict length ({len(self.CTLABELS)}) must match VOC_SIZE ({self.voc_size_config})")
        else:
             raise ValueError(f"Unsupported voc_size_config ({self.voc_size_config}) & no custom dict.")
        self.voc_size = len(self.CTLABELS)
        if not self.use_customer_dict and self.voc_size_config > self.voc_size:
            self.logger.warning(f"Config VOC_SIZE ({self.voc_size_config}) > CTLABELS length ({self.voc_size}). "
                                f"Indices {self.voc_size}-{self.voc_size_config-1} are implicit <unk>.")


    def get_lm_score(self, current_human_readable_text, next_char_idx):
        # ... (Your working "alternative" get_lm_score method) ...
        if self.language_model is None or next_char_idx == self.blank_idx: # self.blank_idx is voc_size_config
            return 0.0
        if next_char_idx >= self.voc_size: # self.voc_size is len(CTLABELS)
            return 0.0 # This is an <unk> from model's perspective relative to CTLABELS

        try:
            next_char_for_lm = self.CTLABELS[next_char_idx]
            if isinstance(next_char_for_lm, int): next_char_for_lm = chr(next_char_for_lm)
            
            context_tokens = list(current_human_readable_text) if current_human_readable_text else []
            full_sequence_tokens = context_tokens + [next_char_for_lm]
            sentence_to_score_full = " ".join(full_sequence_tokens)
            
            log10_prob_full_sequence = self.language_model.score(sentence_to_score_full, bos=True, eos=False)
            log10_prob_context_only = 0.0
            if current_human_readable_text:
                sentence_to_score_context = " ".join(context_tokens)
                if sentence_to_score_context:
                     log10_prob_context_only = self.language_model.score(sentence_to_score_context, bos=True, eos=False)
            
            log10_conditional_prob = log10_prob_full_sequence - log10_prob_context_only
            return log10_conditional_prob
        except Exception as e:
            self.logger.error(f"KenLM scoring error in get_lm_score: {e}", exc_info=False) # exc_info=False for less verbose
            return 0.0

    def ctc_greedy_decode(self, char_logits, silent=False):
        # ... (Your existing ctc_greedy_decode method - ensure it's correct) ...
        if not silent:
            self.logger.debug(f"Pure Greedy decode for logits shape: {char_logits.shape if hasattr(char_logits, 'shape') else 'N/A'}")
        if isinstance(char_logits, np.ndarray):
            char_logits_tensor = torch.from_numpy(char_logits)
        else:
            char_logits_tensor = char_logits
        rec_idx = torch.argmax(char_logits_tensor, dim=-1).numpy()
        last_char_idx = -1
        s = ''
        for c_idx_val in rec_idx:
            c_idx = int(c_idx_val)
            if c_idx == self.blank_idx:
                last_char_idx = -1
            elif c_idx < self.voc_size: # Mappable character
                if last_char_idx != c_idx:
                    try:
                        char = self.CTLABELS[c_idx]
                        if isinstance(char, int): s += chr(char)
                        else: s += char
                        last_char_idx = c_idx
                    except IndexError:
                        if not silent: self.logger.error(f"Greedy Error: Index {c_idx} for CTLABELS (size {self.voc_size})")
                        last_char_idx = -1
            # else: c_idx is implicit <unk> (between self.voc_size and self.blank_idx-1) or truly out of bounds
            elif c_idx < self.blank_idx : # Implicit <unk>
                last_char_idx = -1 # Treat as blank for sequence building
            else: # c_idx > self.blank_idx (should not happen)
                if not silent: self.logger.warning(f"Greedy: Index {c_idx} unexpected (blank={self.blank_idx}).")
                last_char_idx = -1
        return s

    # +++ NEW METHOD: Greedy Decoding with LM influence at each step +++
    def ctc_greedy_decode_with_lm(self, char_logits, silent=False):
        if not silent:
            self.logger.info(f"Greedy decode WITH LM for logits shape: {char_logits.shape if hasattr(char_logits, 'shape') else 'N/A'}")

        if isinstance(char_logits, np.ndarray):
            char_logits_tensor = torch.from_numpy(char_logits).to(self.device)
        else:
            char_logits_tensor = char_logits.to(self.device)

        log_probs_all_steps = F.log_softmax(char_logits_tensor, dim=-1) # T x V_output
        T, V_output = log_probs_all_steps.shape

        decoded_text = ""
        last_emitted_char_idx = -1 # Tracks the last *non-blank* character index added to decoded_text

        for t in range(T):
            log_probs_t = log_probs_all_steps[t] # Log probs for current time step
            best_char_idx_for_step = -1
            max_combined_score_for_step = float('-inf')

            for c_idx_candidate in range(V_output): # Iterate over all possible characters + blank + unk
                log_p_char_vision = log_probs_t[c_idx_candidate].item()
                
                log_p_char_lm = 0.0
                # Get LM score only if it's a character in CTLABELS (not blank, not implicit <unk>)
                if self.language_model and self.lm_weight > 1e-9 and c_idx_candidate < self.voc_size:
                    log_p_char_lm = self.get_lm_score(decoded_text, c_idx_candidate)
                
                current_combined_score = log_p_char_vision + self.lm_weight * log_p_char_lm

                if current_combined_score > max_combined_score_for_step:
                    max_combined_score_for_step = current_combined_score
                    best_char_idx_for_step = c_idx_candidate
            
            # Apply CTC rules with the chosen best_char_idx_for_step
            if best_char_idx_for_step == self.blank_idx:
                last_emitted_char_idx = -1
            elif best_char_idx_for_step < self.voc_size: # Actual character in CTLABELS
                if best_char_idx_for_step != last_emitted_char_idx:
                    try:
                        char_str = self.CTLABELS[best_char_idx_for_step]
                        if isinstance(char_str, int): char_str = chr(char_str)
                        decoded_text += char_str
                        last_emitted_char_idx = best_char_idx_for_step
                    except IndexError: # Should be caught by c_idx < self.voc_size
                        if not silent: self.logger.error(f"Greedy+LM Error: Index {best_char_idx_for_step} for CTLABELS (size {self.voc_size})")
                        last_emitted_char_idx = -1
            elif best_char_idx_for_step < self.blank_idx : # Implicit <unk> token
                last_emitted_char_idx = -1 # Treat as blank for sequence building
            # else: index is > blank_idx, should not happen. If it does, treated as blank.

        if not silent: self.logger.info(f"Greedy+LM result: '{decoded_text}'")
        return decoded_text
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def beam_search_decode(self, char_logits):
        # Using print for beam search internal logs as logger might be too slow/verbose here
        print(f"[BEAM_SEARCH] Start for logits shape: {char_logits.shape if hasattr(char_logits, 'shape') else 'N/A'}")

        if isinstance(char_logits, np.ndarray):
            char_logits_tensor = torch.from_numpy(char_logits).to(self.device)
        else:
            char_logits_tensor = char_logits

        log_probs = F.log_softmax(char_logits_tensor, dim=-1)
        T, V_output = log_probs.shape
        current_beams = {("", -1): 0.0}

        greedy_output_for_this_instance = self.ctc_greedy_decode(char_logits_tensor, silent=True)
        TARGET_GREEDY_FOR_PDB = "underpass" # Example, change if needed
        PDB_TRIGGERED_FOR_INSTANCE = (greedy_output_for_this_instance == TARGET_GREEDY_FOR_PDB)

        # if PDB_TRIGGERED_FOR_INSTANCE:
        #     print(f"\n--- [BEAM_SEARCH_PDB] Tracing instance where greedy is '{TARGET_GREEDY_FOR_PDB}' ---")

        for t in range(T):
            next_beams_candidates = {}
            sorted_current_beams_items = sorted(current_beams.items(), key=lambda item: item[1], reverse=True)[:self.beam_width]

            if not sorted_current_beams_items:
                print(f"[BEAM_SEARCH] Warning: No current beams at start of t={t}. Stopping.")
                break

            for (prev_text, prev_last_char_idx), prev_score in sorted_current_beams_items:
                log_probs_t = log_probs[t]
                for c_idx in range(V_output):
                    log_p_char_vision = log_probs_t[c_idx].item()
                    log_p_char_lm = 0.0
                    if self.language_model and self.lm_weight > 1e-9:
                        log_p_char_lm = self.get_lm_score(prev_text, c_idx)
                    current_char_score = log_p_char_vision + self.lm_weight * log_p_char_lm
                    
                    new_text = prev_text
                    new_last_char_idx = prev_last_char_idx
                    if c_idx == self.blank_idx:
                        new_last_char_idx = -1
                    elif c_idx < self.voc_size:
                        if c_idx != prev_last_char_idx:
                            try:
                                char_str = self.CTLABELS[c_idx]
                                if isinstance(char_str, int): char_str = chr(char_str)
                                new_text = prev_text + char_str
                                new_last_char_idx = c_idx
                            except IndexError: continue
                    elif c_idx < self.blank_idx: 
                         new_last_char_idx = -1
                    
                    beam_key = (new_text, new_last_char_idx)
                    new_beam_score = prev_score + current_char_score

                    if beam_key not in next_beams_candidates or new_beam_score > next_beams_candidates[beam_key]:
                        next_beams_candidates[beam_key] = new_beam_score
            
            if not next_beams_candidates:
                print(f"[BEAM_SEARCH] Warning: No new candidates generated at t={t}. Stopping.")
                break

            sorted_next_candidates_items = sorted(next_beams_candidates.items(), key=lambda item: item[1], reverse=True)
            current_beams = dict(sorted_next_candidates_items[:self.beam_width])

            # PDB for specific instance and timesteps (comment out if not needed)
            # if PDB_TRIGGERED_FOR_INSTANCE:
            #     if t == 6 or t == 7 or t == 8:
            #         print(f"\n>>> PDB: GREEDY='{TARGET_GREEDY_FOR_PDB}', End of t={t} <<<")
            #         # ... (print current_beams items) ...
            #         pdb.set_trace()

        if not current_beams:
            print("[BEAM_SEARCH] Warning: Beam search ended with no valid hypotheses after time loop.")
            return ""

        final_text_scores = {}
        for (text_candidate, _), score_candidate in current_beams.items():
            if text_candidate not in final_text_scores or score_candidate > final_text_scores[text_candidate]:
                final_text_scores[text_candidate] = score_candidate
        
        if not final_text_scores:
            print("[BEAM_SEARCH] Warning: final_text_scores is empty.")
            return ""

        # --- This is where best_text and max_norm_score are defined ---
        best_text = ""
        max_norm_score = float('-inf')
        for text, score in final_text_scores.items():
            norm_score = score / (len(text) + 1e-6) if text else float('-inf')
            if norm_score > max_norm_score:
                max_norm_score = norm_score
                best_text = text
        # --- PDB Before Final Length Normalized Selection (ensure it's after best_text is found) ---
        # if PDB_TRIGGERED_FOR_INSTANCE:
        #     print("\n>>> PDB: After final length-normalized selection. <<<")
        #     print(f"    Greedy output for this instance was: '{greedy_output_for_this_instance}'")
        #     print(f"    Selected best_text: '{best_text}' (NormScore: {max_norm_score:.3f})")
        #     print(f"    final_text_scores (top 5 by raw score):")
        #     sorted_final_raw_scores = sorted(final_text_scores.items(), key=lambda item: item[1], reverse=True)
        #     for i, (txt, scr) in enumerate(sorted_final_raw_scores[:min(5, len(sorted_final_raw_scores))]):
        #         norm_s = scr / (len(txt) + 1e-6) if txt else float('-inf')
        #         print(f"      PDB Final Cand {i}: '{txt}' (RawScore: {scr:.3f}, Potential NormScore: {norm_s:.3f})")
        #     pdb.set_trace()
        # --- End PDB ---
        
        print(f"[BEAM_SEARCH] Finished. Best text: '{best_text}', NormScore: {max_norm_score:.3f}, RawScore: {final_text_scores.get(best_text, float('-nan')):.3f}")
        return best_text


    def refine(self, char_logits, instance_score_for_log=None):
        self.logger.info(f"--- refine method ENTERED (logits shape: {char_logits.shape if hasattr(char_logits, 'shape') else 'N/A'}) ---")
        try:
            if isinstance(char_logits, np.ndarray): # Ensure tensor for processing
                char_logits_tensor = torch.from_numpy(char_logits)
            else:
                char_logits_tensor = char_logits

            lm_active = self.language_model is not None and self.lm_weight > 1e-9 and hasattr(self, 'lm_order') and self.lm_order > 0
            
            self.logger.info(f"  Refine conditions: LM Active: {lm_active}, beam_width: {self.beam_width}")

            if lm_active and self.beam_width > 1:
                self.logger.info(f"    >>> Using BEAM SEARCH (beam_width={self.beam_width}, lm_weight={self.lm_weight}, lm_order={self.lm_order})")
                refined_text = self.beam_search_decode(char_logits_tensor)
            elif lm_active and self.beam_width == 1: # Or just lm_active and default to this if beam_width=1
                self.logger.info(f"    >>> Using GREEDY DECODING WITH LM (beam_width={self.beam_width}, lm_weight={self.lm_weight}, lm_order={self.lm_order})")
                refined_text = self.ctc_greedy_decode_with_lm(char_logits_tensor)
            else: # Pure greedy (LM not loaded or lm_weight is effectively zero, or beam_width specified as <1)
                log_reason = f"(LM loaded: {self.language_model is not None}, beam_width: {self.beam_width}, lm_order: {self.lm_order if hasattr(self, 'lm_order') else 'N/A'}, lm_weight: {self.lm_weight})"
                self.logger.info(f"    >>> Using PURE GREEDY DECODING. Reason/State: {log_reason}")
                refined_text = self.ctc_greedy_decode(char_logits_tensor)

            if self.logger.isEnabledFor(logging.INFO):
                greedy_text_for_debug = self.ctc_greedy_decode(char_logits_tensor, silent=True)
                score_str = f"(Score: {instance_score_for_log:.3f})" if instance_score_for_log is not None else ""
                if refined_text != greedy_text_for_debug:
                    self.logger.info(f"    DIFFERENCE! {score_str} PureGreedy: '{greedy_text_for_debug}', Refined: '{refined_text}'")
                else:
                    if self.logger.isEnabledFor(logging.DEBUG): # Only show no difference in DEBUG
                        self.logger.debug(f"    NO DIFFERENCE. {score_str} PureGreedy/Refined: '{refined_text}'")
            return refined_text
        
        except Exception as e:
            self.logger.error(f"!!! EXCEPTION in FULL refine method: {e}", exc_info=True)
            self.logger.error("!!! FALLING BACK TO PURE GREEDY DECODING DUE TO EXCEPTION.")
            return self.ctc_greedy_decode(char_logits_tensor if 'char_logits_tensor' in locals() else char_logits, silent=True)