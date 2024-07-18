import torch
import torchaudio
from seamless_communication.inference import Translator
from datasets import load_dataset
from torch.utils.data import DataLoader
import argparse
from opencc import OpenCC
from jiwer import wer
import os
import re
import json
from tqdm import tqdm

cc = OpenCC('t2s')

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', type=str, default="ASR")
    parser.add_argument('--src_lang', type=str, default='cmn')
    parser.add_argument('--tgt_lang', type=str, default='cmn')
    parser.add_argument('--model_name_or_card', '-m', type=str, default="seamlessM4T_v2_large") # seamlessM4T_large, seamlessM4T_medium
    parser.add_argument('--vocoder_name_or_card', '-v', type=str, default="vocoder_v2") # vocoder_36langs
    parser.add_argument('--output_dir', '-o', type=str, default="./seamless_output")
    parser.add_argument('--exp_name', '-n', type=str, default="results")
    parser.add_argument('--split', '-s', type=str, default="test")
    parser.add_argument('--label', '-b', type=int)
    parser.add_argument('--dataset_path', '-d', type=str, default='CAiRE/ASCEND')
    parser.add_argument('--use_domain_tag', '-u', action='store_true')
    return parser.parse_args()

def insert_space_in_code_switched_text(text):
    text = text.lower()
    # Regular expression to match Chinese characters
    chinese_char_pattern = r'[\u4e00-\u9fff]'

    # Insert space before and after each Chinese character
    spaced_text = re.sub(f'({chinese_char_pattern})', r' \1 ', text)

    # Remove punctuations
    spaced_text = re.sub(r'[^\w\s]', '', spaced_text)
    
    # Remove any extra spaces added by the previous step
    normalized_text = re.sub(r'\s+', ' ', spaced_text)
    normalized_text = normalized_text.strip().replace("  ", " ")
    return normalized_text

def calculate_MER(results):
    hyps = []
    refs = []
    new_results = []
    for result in results:
        p = cc.convert(result["prediction"])
        p = p.strip()
        p = insert_space_in_code_switched_text(p)
        hyps.append(p)

        t = insert_space_in_code_switched_text(cc.convert(result["transcription"]))
        refs.append(t)

        new_results.append({
            "id": result["id"],
            "prediction": p,
            "transcription": t,
            "raw_prediction": result["prediction"],
            "topic": result["topic"]
        })
    return new_results, wer(refs, hyps)


if __name__ == "__main__":
    args = get_args_parser()
    print(args.task)
    assert args.tgt_lang is not None
    if args.task == "ASR":
        if args.src_lang is None:
            args.src_lang = args.tgt_lang
        else:
            assert args.src_lang == args.tgt_lang
    
    if not os.path.exists(args.output_dir):
        print("Create output directory:", args.output_dir)
        os.makedirs(args.output_dir)
    else:
        print("Output directory exists:", args.output_dir)

    DATASET_PATH = args.dataset_path
    dataset = load_dataset(DATASET_PATH, split=args.split, cache_dir="./cache")
    print("="*15, "Dataset Info", "="*15)
    print("Dataset:", DATASET_PATH)

    # spliting topics
    topics = ['education', 'persona', 'technology', 'philosophy', 'sports']
    topic2dataset = {}
    total_size = 0
    for topic in topics:
        topic_dataset = dataset.filter(lambda x: x['topic'] == topic)
        print(f"{topic} dataset size: {topic_dataset.num_rows}")

        if topic_dataset.num_rows > 0:
            topic2dataset[topic] = topic_dataset
            total_size += topic2dataset[topic].num_rows
    
    assert total_size == dataset.num_rows, "The sum of all topics is not equal to the total size of the dataset. Some topics are missing."
    
    # Initialize a Translator object with a multitask model, vocoder on the GPU.
    translator = Translator(args.model_name_or_card, args.vocoder_name_or_card, torch.device("cuda:0"), dtype=torch.float16).eval()

    results = []

    for topic, topic_dataset in topic2dataset.items():
        for _, data in enumerate(tqdm(topic_dataset)):
            audio = torch.from_numpy(data["audio"]['array']).to(torch.device("cuda:0"))
            text_output, _ = translator.predict(
                input=audio, 
                task_str=args.task,
                src_lang=args.src_lang,
                tgt_lang=args.tgt_lang, 
            )

            results.append({
                "id": data['audio']['path'],
                "prediction": str(text_output[0]),
                "transcription": data["transcription"],
                "topic": topic,
            })

    results, word_error_rate = calculate_MER(results)
    print("Number of data:", len(results))
    print("WER:", word_error_rate)

    json.dump(
        {"model_name_or_card": args.model_name_or_card, "vocoder_name_or_card": args.vocoder_name_or_card, "MER": word_error_rate, "results": results}, open(f"{args.output_dir}/{args.exp_name}.json", "w", encoding='utf16'), indent=2, ensure_ascii=False
    )