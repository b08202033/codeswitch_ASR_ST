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
from prompt_whisper import insert_space_in_code_switched_text

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
    parser.add_argument('--dataset_path', '-d', type=str, default='./zh_en')
    return parser.parse_args()

def calculate_MER(results):
    hyps = []
    refs = []
    new_results = []
    for result in results:
        p = cc.convert(result["prediction"])
        p = p.strip()
        p = insert_space_in_code_switched_text(p)
        t = insert_space_in_code_switched_text(cc.convert(result["transcription"]))
        hyps.append(p)
        refs.append(t)
        new_results.append({
            "id": result["id"],
            "prediction": p,
            "transcription": t,
            "raw_prediction": result["prediction"],
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

    if args.label is not None:
        dataset = dataset.filter(lambda x: x['label'] == args.label)
        assert dataset[0]['label'] == args.label
    
    # Initialize a Translator object with a multitask model, vocoder on the GPU.
    translator = Translator(args.model_name_or_card, args.vocoder_name_or_card, torch.device("cuda:0"), dtype=torch.float16).eval()

    results = []
    for _, data in enumerate(tqdm(dataset)):
        audio = torch.from_numpy(data["audio"]['array']).to(torch.device("cuda:0"))
        text_output, _ = translator.predict(
            input=audio, 
            task_str=args.task,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang, 
        )

        file = ''
        if 'label' in dataset.features.keys():
            label = 'correct' if data['label'] == 0 else 'wrong'
            file = f'{label}/' + data['audio']['path']
        else:
            file = data['audio']['path']

        results.append({
            "id": file,
            "prediction": str(text_output[0]),
            "transcription": data["transcription"],
        })

    results, word_error_rate = calculate_MER(results)
    print("Number of data:", len(results))
    print("WER:", word_error_rate)

    json.dump(
        {"model_name_or_card": args.model_name_or_card, "vocoder_name_or_card": args.vocoder_name_or_card, "MER": word_error_rate, "results": results}, open(f"{args.output_dir}/{args.exp_name}.json", "w", encoding='utf16'), indent=2, ensure_ascii=False
    )