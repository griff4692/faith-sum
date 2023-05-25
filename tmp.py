import pandas as pd

# Dendrite
# python gen_from_extract.py --split test --decode_method beam -add_abstract_experiment --extract_experiment cnn_e_final --abstract_experiment cnn_ea_final_mle_0.1_like_10.0_unlike_10.0 --num_candidates 16 --num_return_sequences 1 --device 0 --chunk 0
# python gen_from_extract.py --split test --decode_method beam -add_abstract_experiment --extract_experiment cnn_e_final --abstract_experiment cnn_ea_final_mle_0.1_like_10.0_unlike_10.0 --num_candidates 16 --num_return_sequences 1 --device 0 --chunk 1
# python gen_from_extract.py --split test --decode_method beam -add_abstract_experiment --extract_experiment cnn_e_final --abstract_experiment cnn_ea_final_mle_0.1_like_10.0_unlike_10.0 --num_candidates 16 --num_return_sequences 1 --device 0 --chunk 2
# python gen_from_extract.py --split test --decode_method beam -add_abstract_experiment --extract_experiment cnn_e_final --abstract_experiment cnn_ea_final_mle_0.1_like_10.0_unlike_10.0 --num_candidates 16 --num_return_sequences 1 --device 0 --chunk 3

# Dendrite
# python gen_from_extract.py --split test --decode_method beam -add_abstract_experiment --extract_experiment cnn_e_final --abstract_experiment cnn_ea_final_mle_0.1_like_10.0_unlike_10.0 --num_candidates 16 --num_return_sequences 1 --device 0 --chunk 4
# python gen_from_extract.py --split test --decode_method beam -add_abstract_experiment --extract_experiment cnn_e_final --abstract_experiment cnn_ea_final_mle_0.1_like_10.0_unlike_10.0 --num_candidates 16 --num_return_sequences 1 --device 1 -chunk 5
# python gen_from_extract.py --split test --decode_method beam -add_abstract_experiment --extract_experiment cnn_e_final --abstract_experiment cnn_ea_final_mle_0.1_like_10.0_unlike_10.0 --num_candidates 16 --num_return_sequences 1 --device 2 --chunk 6
# python gen_from_extract.py --split test --decode_method beam -add_abstract_experiment --extract_experiment cnn_e_final --abstract_experiment cnn_ea_final_mle_0.1_like_10.0_unlike_10.0 --num_candidates 16 --num_return_sequences 1 --device 3 --chunk 7


def split_dataframe(df, num_parts):
    total_rows = len(df)
    chunk_size = total_rows // num_parts
    remainder = total_rows % num_parts

    chunks = []
    start = 0

    for i in range(num_parts):
        end = start + chunk_size
        if i < remainder:
            end += 1
        chunks.append(df.iloc[start:end])
        start = end

    return chunks


if __name__ == '__main__':
    df = pd.read_csv('/nlp/projects/faithsum/results/cnn_e_final/test_beam_16_outputs.csv')

    chunks = split_dataframe(df, 8)

    for chunk_idx, chunk in enumerate(chunks):
        chunk.to_csv(f'/nlp/projects/faithsum/results/cnn_e_final/test_beam_16_outputs_chunk_{chunk_idx}.csv')
