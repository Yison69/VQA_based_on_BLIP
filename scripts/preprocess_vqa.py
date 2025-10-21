import json
import os

def preprocess_vqa_data(question_file, annotation_file, output_file):
    """
    Merges VQA questions and annotations into a single file that LAVIS expects.
    """
    print(f"Loading questions from: {question_file}")
    with open(question_file, 'r') as f:
        questions = json.load(f)['questions']

    print(f"Loading annotations from: {annotation_file}")
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)['annotations']

    # Create a dictionary to map question_id to question text
    question_dict = {q['question_id']: q for q in questions}

    merged_data = []
    for ann in annotations:
        question_id = ann['question_id']
        if question_id in question_dict:
            # Merge question and annotation info
            merged_entry = {**question_dict[question_id], **ann}
            merged_data.append(merged_entry)

    print(f"Saving {len(merged_data)} merged entries to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(merged_data, f)
    print("Done.\n")

if __name__ == "__main__":
    # Define the base path to your dataset
    base_path = "/root/autodl-tmp/VQA_BLIP/VQA_based_on_BLIP/dataset/vqav2"

    # --- Process Training Data ---
    train_q_file = os.path.join(base_path, "v2_OpenEnded_mscoco_train2014_questions.json")
    train_a_file = os.path.join(base_path, "v2_mscoco_train2014_annotations.json")
    output_train_file = os.path.join(base_path, "vqa_train_merged.json")
    preprocess_vqa_data(train_q_file, train_a_file, output_train_file)

    # --- Process Validation Data ---
    val_q_file = os.path.join(base_path, "v2_OpenEnded_mscoco_val2014_questions.json")
    val_a_file = os.path.join(base_path, "v2_mscoco_val2014_annotations.json")
    output_val_file = os.path.join(base_path, "vqa_val_merged.json")
    preprocess_vqa_data(val_q_file, val_a_file, output_val_file)
