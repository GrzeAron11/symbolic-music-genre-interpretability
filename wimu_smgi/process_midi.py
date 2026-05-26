from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from loguru import logger
import numpy as np
import pretty_midi
from tqdm import tqdm
import typer

from wimu_smgi.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

MAX_STEPS = 60000


def process_midi(midi_path: Path, fs: int = 100):
    midi_data = pretty_midi.PrettyMIDI(str(midi_path))
    melodic_tracks, drum_tracks = [], []

    min_len = 2.0 / fs

    for instrument in midi_data.instruments:

        if instrument.is_drum:
            # pretty_midi ma problem z robieniem pianorolla z perkusji
            instrument.is_drum = False

            for note in instrument.notes:
                if note.end - note.start < min_len:
                    note.end = note.start + min_len

            roll = instrument.get_piano_roll(fs=fs).astype(np.float32)

            instrument.is_drum = True

            if roll.shape[1] > MAX_STEPS:
                roll = roll[:, :MAX_STEPS]
            drum_tracks.append(roll)

        else:
            roll = instrument.get_piano_roll(fs=fs).astype(np.float32)

            if roll.shape[1] > MAX_STEPS:
                roll = roll[:, :MAX_STEPS]
            melodic_tracks.append(roll)

    def combine_and_transpose(tracks):
        if not tracks:
            return None
        max_len = max(track.shape[1] for track in tracks)
        combined = np.zeros((128, max_len), dtype=np.float32)

        for track in tracks:
            combined[:, :track.shape[1]] += track

        return np.clip(combined, 0, 127).astype(np.uint8).T

    melodic_t = combine_and_transpose(melodic_tracks)
    drum_t = combine_and_transpose(drum_tracks)

    if melodic_t is None and drum_t is None:
        raise ValueError("Plik nie zawiera żadnych poprawnych ścieżek melodycznych ani perkusyjnych.")

    len_m = melodic_t.shape[0] if melodic_t is not None else 0
    len_d = drum_t.shape[0] if drum_t is not None else 0
    max_len = max(len_m, len_d)

    def pad_to_max_len(arr, target_len):
        if arr is None:
            return np.zeros((target_len, 128), dtype=np.uint8)
        if arr.shape[0] < target_len:
            padded = np.zeros((target_len, 128), dtype=np.uint8)
            padded[:arr.shape[0], :] = arr
            return padded
        return arr

    melodic_t = pad_to_max_len(melodic_t, max_len)
    drum_t = pad_to_max_len(drum_t, max_len)

    return melodic_t, drum_t


def process_single_file(midi_path: Path, output_path: Path, sample_rate: int):
    try:
        melody, drums = process_midi(midi_path, fs=sample_rate)

        track_id = midi_path.parent.name
        short_hash = midi_path.stem[:5]
        out_file = output_path / f"{track_id}__{short_hash}.npz"

        np.savez_compressed(out_file, melody=melody, drums=drums)
        return True, midi_path.name, None
    except Exception as e:
        return False, midi_path.name, str(e)


@app.command()
def main(
    input_path: Path = typer.Option(RAW_DATA_DIR / "lmd_matched", help="Katalog z plikami MIDI"),
    output_path: Path = typer.Option(PROCESSED_DATA_DIR / "npy_arrays", help="Katalog wyjściowy"),
    sample_rate: int = typer.Option(100, "--sample-rate", help="Rozdzielczość w Hz"),
    workers: int = typer.Option(5, "--workers", help="Liczba rdzeni procesora do użycia (np. 4, 8)"),
):
    logger.info(f"Starting MIDI files processing... {input_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    midi_files = list(input_path.rglob("*.mid")) + list(input_path.rglob("*.midi"))

    if not midi_files:
        logger.warning(f"No midi files found in {input_path}!")
        raise typer.Exit()

    logger.info(f"Found {len(midi_files)} MIDI files. Starting multiprocessing with {workers} workers.")

    success_count = 0
    fail_count = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:

        futures = {
            executor.submit(process_single_file, path, output_path, sample_rate): path
            for path in midi_files
        }

        for future in tqdm(as_completed(futures), total=len(midi_files), desc="Processing dataset"):
            success, filename, error_msg = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
                logger.error(f"Failed to process {filename}: {error_msg}")

    logger.success(f"Processing complete! Success: {success_count}, Failed: {fail_count}. Saved to {output_path}")


if __name__ == "__main__":
    app()
