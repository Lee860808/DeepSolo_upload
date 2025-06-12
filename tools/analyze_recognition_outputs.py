# tools/analyze_recognition_outputs.py
import json
import os
import argparse
import yaml # For parsing config file

# It's good practice to have editdistance available if you want to do
# more advanced correctness checks later, but not strictly needed for exact match.
# import editdistance 

def load_json_log(filepath):
    """Loads data from a JSON-lines log file."""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f: # Added encoding
            for line_num, line in enumerate(f):
                line_content = line.strip()
                if not line_content: # Skip empty lines
                    continue
                try:
                    # Skip common non-JSON log/debug lines if they accidentally get mixed in
                    if line_content.startswith("[") or \
                       line_content.startswith("DEBUG:") or \
                       line_content.startswith("WARNING:") or \
                       line_content.startswith("ERROR:") or \
                       "DIFFERENCE DETECTED" in line_content or \
                       "---" in line_content or ">>>" in line_content:
                        # print(f"Skipping non-JSON line #{line_num+1} in {filepath}: {line_content}")
                        continue
                    data.append(json.loads(line_content))
                except json.JSONDecodeError:
                    print(f"Skipping malformed JSON line #{line_num+1} in {filepath}: {line_content}")
    except FileNotFoundError:
        print(f"Error: Log file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading log file {filepath}: {e}")
        return None
    return data

def load_gt_instances_for_image(gt_files_dir, image_id_from_log, dataset_name_short):
    """
    Loads GT text instances for a single image from its .txt file.
    Adapts parsing based on dataset_name_short.
    `image_id_from_log` is the ID from your prediction logs (e.g., from COCO JSON).
    """
    gt_filename = ""
    # --- CRITICAL: Adjust filename construction based on your GT file naming ---
    if "ic15" in dataset_name_short:
        # Common IC15 GT naming (e.g., for image ID 1, file is "gt_img_1.txt")
        # This assumes image_id_from_log is 1-based. If it's 0-based from D2, adjust.
        # Example: if image_id_from_log '2' refers to the file named 'gt_img_2.txt'
        gt_filename = f"gt_img_{str(image_id_from_log)}.txt"
        # If your IC15 GT files are just "1.txt", "2.txt", etc. use:
        # gt_filename = f"{str(image_id_from_log)}.txt"
    elif "totaltext" in dataset_name_short or "ctw1500" in dataset_name_short:
        # Assuming your standardized GT files for these are simply {image_id}.txt
        # and image_id_from_log is the correct numeric string.
        gt_filename = f"{str(image_id_from_log)}.txt"
    else:
        print(f"Warning: Unknown dataset '{dataset_name_short}' for GT filename construction. Assuming '{str(image_id_from_log)}.txt'.")
        gt_filename = f"{str(image_id_from_log)}.txt"
    # --- END CRITICAL ADJUSTMENT ---

    gt_filepath = os.path.join(gt_files_dir, gt_filename)
    
    gt_instances_texts = []
    if not os.path.exists(gt_filepath):
        # This warning is useful
        # print(f"Warning: GT file not found for image_id {image_id_from_log} (tried {gt_filename}) in dir {gt_files_dir}")
        return [] 

    try:
        with open(gt_filepath, 'r', encoding='utf-8-sig') as f_gt: # utf-8-sig handles BOM
            lines = f_gt.readlines()

            # Handle CTW1500-like format with count on the first line
            if "ctw1500" in dataset_name_short and lines:
                try:
                    int(lines[0].strip()) # Check if first line is a count
                    lines = lines[1:]     # Skip count line
                except ValueError:
                    pass # First line is not a count, process all lines

            for line_num, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                gt_text = "GT_PARSE_ERROR"
                if "ic15" in dataset_name_short:
                    if line.startswith('\ufeff'): line = line[1:] # Remove BOM if present
                    parts = line.split(',')
                    if len(parts) >= 9: # 8 coords + at least 1 part for transcription
                        transcription_parts = parts[8:]
                        gt_text = ",".join(transcription_parts).strip()
                        if gt_text.startswith('"') and gt_text.endswith('"'):
                            gt_text = gt_text[1:-1]
                        gt_text = gt_text.replace('""','"') # Handle double quotes inside if they are escaped this way
                    else:
                        gt_text = "GT_MALFORMED_IC15_LINE"
                
                # For TotalText and CTW1500 (after skipping count for CTW) - standardized format
                elif "totaltext" in dataset_name_short or "ctw1500" in dataset_name_short:
                    parts = line.split(',####')
                    if len(parts) == 2:
                        gt_text = parts[1].strip()
                    # Handle case where GT line might be just "###" without coords and separator
                    elif len(parts) == 1 and parts[0].strip() == "###":
                        gt_text = "###"
                    else:
                        gt_text = "GT_MALFORMED_STD_LINE"
                else: # Fallback for unknown dataset types
                    gt_text = line # Assume whole line is text if format is unknown

                gt_instances_texts.append(gt_text)
        return gt_instances_texts
    except Exception as e:
        print(f"Error reading or parsing GT file {gt_filepath}: {e}")
        return ["GT_FILE_READ_ERROR"] * len(lines) # Return placeholders if file read fails

def analyze_outputs(config_beam_width, config_lm_weight, config_lm_path_used,
                    lm_difference_log_path, gt_files_dir, output_analysis_log_path,
                    dataset_name_short):
    
    logged_predictions = load_json_log(lm_difference_log_path)
    if logged_predictions is None:
        print(f"No data loaded from {lm_difference_log_path}. Exiting analysis.")
        return

    lm_actually_loaded = bool(config_lm_path_used and os.path.exists(config_lm_path_used))
        
    decoding_method_used = "UNKNOWN"
    if lm_actually_loaded and config_lm_weight > 1e-9:
        if config_beam_width > 1:
            decoding_method_used = "BeamSearch+LM"
        elif config_beam_width == 1:
            decoding_method_used = "Greedy+LM"
        else:
             decoding_method_used = "PureGreedy_VisionOnly_FallbackBeamWidth" # Beamwidth < 1 or invalid
    else: 
        decoding_method_used = "PureGreedy_VisionOnly"

    print(f"Analysis based on assumed effective decoding method for the run: {decoding_method_used}")
    print(f"(Inferred from config: BeamWidth={config_beam_width}, LM_Weight={config_lm_weight}, LM_Path_Used='{config_lm_path_used}')")

    analysis_results = []
    correct_pure_greedy_count = 0
    correct_refined_count = 0
    total_instances_compared = 0

    preds_by_image = {}
    for pred_entry in logged_predictions:
        img_id = pred_entry.get("image_id")
        if img_id is None:
            print(f"Warning: Prediction entry missing 'image_id': {pred_entry}")
            continue
        if img_id not in preds_by_image:
            preds_by_image[img_id] = []
        preds_by_image[img_id].append(pred_entry)

    for image_id, predictions_for_image in preds_by_image.items():
        gt_texts_for_image = load_gt_instances_for_image(gt_files_dir, image_id, dataset_name_short)
        
        if not gt_texts_for_image and any(p.get("pure_greedy_text") or p.get("lm_refined_text") for p in predictions_for_image):
            print(f"Warning: No GT instances found or loaded for image_id {image_id} (GT file: {os.path.join(gt_files_dir, f'{image_id}.txt' or f'gt_img_{image_id}.txt')}), but predictions exist.")

        for pred_entry in predictions_for_image:
            instance_idx = pred_entry.get("instance_index")
            pure_greedy_text = pred_entry.get("pure_greedy_text", "")
            refined_text = pred_entry.get("lm_refined_text", "")
            detection_score = pred_entry.get("score", 0.0)

            gt_text_for_this_pred = "GT_INSTANCE_NOT_FOUND_OR_INDEX_OOB"
            if instance_idx is not None and instance_idx < len(gt_texts_for_image):
                gt_text_for_this_pred = gt_texts_for_image[instance_idx]
            
            # Normalize for comparison (case-insensitive)
            pure_greedy_norm = pure_greedy_text.lower()
            refined_norm = refined_text.lower()
            gt_norm = gt_text_for_this_pred.lower()

            # Exclude "don't care" GTs from accuracy calculation, but still log them
            is_gt_dont_care = (gt_text_for_this_pred == "###")
            
            is_pure_greedy_correct = False
            is_refined_correct = False

            if not is_gt_dont_care and "GT_" not in gt_text_for_this_pred: # Only compare if GT is valid
                is_pure_greedy_correct = (pure_greedy_norm == gt_norm)
                is_refined_correct = (refined_norm == gt_norm)
                total_instances_compared +=1 # Count only if GT is valid for comparison
                if is_pure_greedy_correct:
                    correct_pure_greedy_count +=1
                if is_refined_correct:
                    correct_refined_count +=1
            
            analysis_entry = {
                "image_id": image_id,
                "instance_idx_pred": instance_idx,
                "detection_score": detection_score,
                "ground_truth": gt_text_for_this_pred,
                "pure_greedy_pred": pure_greedy_text, # Log original case
                "pure_greedy_correct": is_pure_greedy_correct,
                "final_refined_pred": refined_text,    # Log original case
                "refined_pred_correct": is_refined_correct,
                "decoding_method_applied": decoding_method_used
            }
            analysis_results.append(analysis_entry)

    # Save the detailed analysis log
    try:
        with open(output_analysis_log_path, 'w', encoding='utf-8') as f_out:
            for entry in analysis_results:
                f_out.write(json.dumps(entry) + "\n")
        print(f"Detailed analysis log saved to: {output_analysis_log_path}")
    except Exception as e:
        print(f"Error writing output analysis log to {output_analysis_log_path}: {e}")


    if total_instances_compared > 0:
        print(f"\nSummary (based on {total_instances_compared} instances with valid GT):")
        print(f"Pure Greedy Accuracy: {correct_pure_greedy_count / total_instances_compared:.4f} ({correct_pure_greedy_count}/{total_instances_compared})")
        print(f"Refined (by {decoding_method_used}) Accuracy: {correct_refined_count / total_instances_compared:.4f} ({correct_refined_count}/{total_instances_compared})")
    else:
        print("\nSummary: No valid instances were compared (e.g., all GTs missing or marked as 'don't care').")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Text Spotting Recognition Outputs")
    parser.add_argument("--config_file", required=True, help="Path to the D2 config YAML file used for the run that generated the diff_log.")
    parser.add_argument("--diff_log", required=True, help="Path to the lm_difference_analysis_{dataset}.log file.")
    parser.add_argument("--gt_dir", required=True, help="Path to the directory containing individual GT .txt files.")
    parser.add_argument("--output_log", required=True, help="Path to save the new detailed analysis log.")
    parser.add_argument("--dataset_name", required=True, choices=['ic15', 'totaltext', 'ctw1500', 'rects'], # Added rects as an example
                        help="Short name of the dataset being analyzed (for GT parsing and filename conventions).")
    
    args = parser.parse_args()
    
    cfg_beam_width = 1 
    cfg_lm_weight = 0.0
    cfg_lm_path = "" # Intended LM path from config
    try:
        with open(args.config_file, 'r') as f_cfg:
            config_yaml = yaml.safe_load(f_cfg)
        if 'MODEL' in config_yaml and 'POST_PROCESS' in config_yaml['MODEL']:
            post_process_section = config_yaml['MODEL']['POST_PROCESS']
            cfg_beam_width = post_process_section.get('BEAM_WIDTH', 1)
            cfg_lm_weight = post_process_section.get('LM_WEIGHT', 0.0)
            cfg_lm_path = post_process_section.get('LM_PATH', "")
    except Exception as e:
        print(f"Warning: Could not parse BEAM_WIDTH/LM_WEIGHT/LM_PATH from config {args.config_file}: {e}. Using defaults.")

    analyze_outputs(
        cfg_beam_width,
        cfg_lm_weight,
        cfg_lm_path, # Pass the path that was configured for the run
        args.diff_log,
        args.gt_dir,
        args.output_log,
        args.dataset_name
    )
