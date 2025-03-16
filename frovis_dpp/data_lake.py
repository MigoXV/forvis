from itertools import chain
from pathlib import Path

import pandas as pd


def scan_dir(input_dir: Path, organ: str):
    """
    Scans the given directory for image files and returns a DataFrame with image paths and organ labels.

    Parameters:
        input_dir (Path): The directory to scan for image files.
        organ (str): The label for the organ associated with the images.

    Returns:
        pd.DataFrame: A DataFrame with two columns: 'image' (file path as string) and 'organ' (organ label).
    """
    image_list = chain(
        input_dir.glob("**/*.jpg"),
        input_dir.glob("**/*.png"),
        input_dir.glob("**/*.bmp"),
    )
    image_list = list(image_list)
    image_df = pd.DataFrame(
        {"image": [str(img) for img in image_list], "organ": [organ] * len(image_list)}
    )
    return image_df


if __name__ == "__main__":
    input_dirs = ["data-bin/数据集/lung/images", "data-bin/数据集/肾脏", "data-bin/心"]
    input_dirs = [Path(path) for path in input_dirs]
    organs = ["lung", "kidney", "heart"]
    output_dir = Path("data-bin/organ")
    df_list = []
    for input_dir, organ in zip(input_dirs, organs):
        df = scan_dir(input_dir, organ)
        df.to_csv(output_dir / f"{organ}.tsv", index=False, sep="\t")
        df_list.append(df)
    final_df = pd.concat(df_list, ignore_index=True)
    final_df.to_csv(output_dir / "all_organs.tsv", index=False, sep="\t")
    # # Combine all dataframes into one
    # df_list = [scan_dir(input_dir, organ) for input_dir, organ in zip(input_dirs, organs)]
    # final_df = pd.concat(df_list, ignore_index=True)

    # # Display or save the DataFrame
    # print(final_df)
    # final_df.to_csv("data-bin/organ/e01.tsv", index=False, sep="\t")
