from pathlib import Path
import pandas as pd
from loguru import logger
import typer

from wimu_smgi.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_file: str = typer.Argument(
        "labels_raw.cls", 
        help="Labels file in .cls format"
    ),
    output_file: str = typer.Option(
        "dataset.csv", 
        "--out", "-o", 
        help="Output CSV file name (default: dataset.csv)"
    ),
):
    input_path = RAW_DATA_DIR / "labels" / input_file
    output_path = PROCESSED_DATA_DIR / "labels" / output_file

    logger.info(f"Starting generation of One-Hot CSV from file: {input_path}")

    if not input_path.exists():
        logger.error(f"File not found: {input_path}")
        raise typer.Exit(code=1)

    try:
        df = pd.read_csv(input_path, sep='\t', header=None, names=['file_id', 'genre'])
        logger.info(f"Loaded {len(df)} rows of raw labels.")
        one_hot_df = pd.crosstab(df['file_id'], df['genre']).clip(upper=1).reset_index()

        genre_list_df = df.groupby('file_id')['genre'].apply(lambda x: list(set(x))).reset_index(name='genre_list')
        final_df = pd.merge(one_hot_df, genre_list_df, on='file_id')

        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_path, index=False)

        logger.info(f"Identified genres (columns): {list(one_hot_df.columns)[1:]}")
        logger.success(f"CSV file saved successfully: {output_path}")

    except Exception as e:
        logger.error(f"Error occurred while creating CSV: {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()