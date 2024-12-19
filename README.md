# Investigating Zero-Shot Generalizability on Mandarin-English Code-Switched ASR and Speech-to-text Translation of Recent Foundation Models with Self-Supervision and Weak Supervision
This is the implementation of the work "Investigating Zero-Shot Generalizability on Mandarin-English Code-Switched ASR and Speech-to-text Translation of Recent Foundation Models with Self-Supervision and Weak Supervision"

## Datasets
All the used datasets are publicly available. Here's the links:
- ASCEND: https://huggingface.co/datasets/CAiRE/ASCEND
- CSZS-correct-zh: https://huggingface.co/datasets/ky552/cszs_zh_en
- NTUML2021: https://huggingface.co/datasets/ky552/ML2021_ASR_ST    
Especially, NTUML2021 is a novel dataset proposed in this work. It contains high-quality recording of machine learning course in National Taiwan University, which can serve as code-switched resources and domain-specific corpus.

## Usage
You can run the codes by a simple command:
```
python prompt_whisper.py -t transcribe -l zh -m "openai/whisper-large-v3" \\
-o test_output -n results -c "<|zh|><|en|><|transcribe|><|notimestamps|>" \\
-s test -d ky552/cszs_zh_en
```
and for ASCEND:
```
python prompt_whisper_ASCEND.py -t transcribe -l zh -m "openai/whisper-large-v3" \\
-o test_output -n results -c "<|zh|><|en|><|transcribe|><|notimestamps|>" \\
-s test -d CAiRE/ASCEND
```
In the above examples, ``-t``, ``-l``, ``-m``, ``-c``, ``-s``, ``-d`` specify the task, language, model, the overwrite tokens, split and dataset.

The usage of Seamless is quite similar.

## Citations
If you find this repo helpful, please consider to cite our works as:
```
@INPROCEEDINGS{10446737,
  author={Huang, Kuan-Po and Yang, Chih-Kai and Fu, Yu-Kuan and Dunbar, Ewan and Lee, Hung-Yi},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Zero Resource Code-Switched Speech Benchmark Using Speech Utterance Pairs for Multiple Spoken Languages}, 
  year={2024},
  volume={},
  number={},
  pages={10006-10010},
  keywords={Speech coding;Benchmark testing;Signal processing;Linguistics;Acoustics;Speech processing;Task analysis;Code-switch;Multilingual;Discrete unit;Zero resource;Self-supervised},
  doi={10.1109/ICASSP48485.2024.10446737}}

@inproceedings{yang2024investigating,
  title={Investigating zero-shot generalizability on mandarin-english code-switched asr and speech-to-text translation of recent foundation models with self-supervision and weak supervision},
  author={Yang, Chih-Kai and Huang, Kuan-Po and Lu, Ke-Han and Kuan, Chun-Yi and Hsiao, Chi-Yuan and Lee, Hung-yi},
  booktitle={2024 IEEE International Conference on Acoustics, Speech, and Signal Processing Workshops (ICASSPW)},
  pages={540--544},
  year={2024},
  organization={IEEE}
}
```
