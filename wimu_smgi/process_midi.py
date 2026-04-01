from pathlib import Path

from loguru import logger
from tqdm import tqdm
import numpy as np
import typer
import pretty_midi
from wimu_smgi.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def process_midi(midi_path: Path, fs: int = 100):
    midi_data = pretty_midi.PrettyMIDI(str(midi_path))
    melodic_tracks, drum_tracks = [], [] #podzial na melodie i perkusje jak w paperze Devakosa
    for instrument in midi_data.instruments:
        roll = instrument.get_piano_roll(fs=fs)
        if instrument.is_drum:
            drum_tracks.append(roll)
        else:
            melodic_tracks.append(roll)

    def combine_and_transpose(tracks):
        if not tracks:
            return None
        max_len = max(track.shape[1] for track in tracks)
        combined = np.zeros((128, max_len))
        for track in tracks:
            combined[:, :track.shape[1]] += track
        combined = np.clip(combined, 0, 127)
        return combined.T
    
    melodic_t = combine_and_transpose(melodic_tracks)
    drum_t = combine_and_transpose(drum_tracks)
    if drum_t is None and melodic_t is not None:
        drum_t = np.zeros_like(melodic_t)
    elif melodic_t is None and drum_t is not None:
        melodic_t = np.zeros_like(drum_t)

    return melodic_t, drum_t


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "lmd_matched",
    output_path: Path = PROCESSED_DATA_DIR / "npy_arrays",
):
    logger.info(f"Starting MIDI files processing... {input_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    midi_files = list(input_path.rglob("*.mid")) + list(input_path.rglob("*.midi"))
    if not midi_files:
        logger.warning(f"No midi files found in {input_path}!")
        raise typer.Exit()
    logger.info(f"Found {len(midi_files)} MIDI files.")
    for midi_path in tqdm(midi_files, desc="Processing dataset"):
        try:
            melody, drums = process_midi(midi_path)
            
            #w folderach jest kilka wersji midi tego samego utworu, id utworu to nazwa folderu, a nie nazwa pliku
            track_id = midi_path.parent.name
            short_hash = midi_path.stem[:5]
            out_file = output_path / f"{track_id}__{short_hash}.npy"
            np.save(out_file, {'melody': melody, 'drums': drums})
        except Exception as e:
            logger.error(f"Failed to process {midi_path.name}: {e}")

    logger.success(f"Dataset processing complete! .npy files saved to {output_path}")

if __name__ == "__main__":
    app()
