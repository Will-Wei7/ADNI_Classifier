import os
import glob
import json
import argparse
import pyarrow as pa
import transformers


def _format_text(text, bos_token, eos_token):
    try:
        while text[0] == bos_token: text = text[1:]
        while text[-1] == eos_token: text = text[:-1]
        return text
    except:
        print(f"Format error: {text}")
        return []


def main(args):
    file_path_list = glob.glob(os.path.join(args.data_path, '*.chunk.*.jsonl'))
    num_files = len(file_path_list)
    file_idx = args.rank % num_files
    file_path = file_path_list[file_idx]

    assert args.world_size % num_files == 0, 'world_size must be divisible by the number of files'
    offset = args.rank // num_files
    num_ranks_per_file = args.world_size // num_files

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)

    schema = pa.schema([pa.field('input_ids', pa.uint32())])

    with pa.ipc.new_file(os.path.join(args.output_path, f"rank_{args.rank}.arrow"), schema) as writer:
        with open(file_path, 'r') as file:
            current_line = 0
            num_tokens = 0
            while line := file.readline():
                if current_line % num_ranks_per_file == offset:
                    text = json.loads(line)['text']
                    tokens = tokenizer(text)['input_ids']
                    tokens = _format_text(tokens, tokenizer.bos_token_id, tokenizer.eos_token_id)
                    if tokens:
                        writer.write(pa.record_batch([tokens], schema=schema))
                        num_tokens += len(tokens)
                current_line += 1
                if num_tokens >= args.max_num_tokens:
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data for training.')
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world_size', type=int, default=2048)

    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)

    parser.add_argument('--max_num_tokens', type=int, default=204_800_000)

    parser.add_argument('--tokenizer', type=str, default='meta-llama/Llama-2-7b-hf')
    args = parser.parse_args()
    main(args)