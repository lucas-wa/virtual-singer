"""UI de demonstração do cantor virtual.

Fluxo na interface:
  1. Gravar/enviar amostras da sua voz  ->  treinar modelo de timbre
  2. Escolher uma partitura (pasta em data/songs/)
  3. Sintetizar: a sua voz cantando a partitura

Rode:  python app/gradio_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr  # noqa: E402

from src import paths  # noqa: E402
from src.hardware import detect_profile  # noqa: E402
from src.pipeline import run as run_pipeline  # noqa: E402
from src.voice import VoiceModel, train_voice  # noqa: E402


def list_songs() -> list[str]:
    if not paths.SONGS.exists():
        return []
    return [d.name for d in sorted(paths.SONGS.iterdir()) if d.is_dir()]


def list_voices() -> list[str]:
    vd = paths.MODELS / "voices"
    if not vd.exists():
        return []
    return [d.name for d in sorted(vd.iterdir())
            if d.is_dir() and VoiceModel.for_name(d.name).exists()]


def do_train(name: str, files: list[str], epochs: int) -> str:
    if not name.strip():
        return "Informe um nome para a voz."
    if not files:
        return "Envie pelo menos um arquivo de áudio (>= 5 min no total)."
    voice_dir = paths.VOICES / name.strip()
    voice_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        dst = voice_dir / Path(f).name
        dst.write_bytes(Path(f).read_bytes())
    model = train_voice(voice_dir, name=name.strip(), epochs=int(epochs))
    return f"Modelo '{name}' treinado: {model.weight}"


def do_synthesize(song: str, voice: str, instrumental: str | None, transpose: int):
    if not song or not voice:
        return None, "Escolha uma música e uma voz."
    out = paths.OUT / f"{voice}_{song}.wav"
    run_pipeline(paths.SONGS / song, voice, out,
                 instrumental=instrumental or None, transpose=int(transpose))
    return str(out), f"Pronto: {out}"


def build_ui() -> gr.Blocks:
    profile = detect_profile()
    with gr.Blocks(title="Cantor Virtual — SVS") as demo:
        gr.Markdown(
            "# 🎤 Cantor Virtual (SVS)\n"
            "Grave a **sua** voz e ouça-a cantando qualquer partitura.\n\n"
            f"**Hardware:** {profile.summary()}\n\n"
            "> Use apenas vozes próprias ou com consentimento — veja CONSENT.md."
        )

        with gr.Tab("1. Treinar minha voz"):
            name = gr.Textbox(label="Nome da voz", placeholder="meu_nome")
            files = gr.File(label="Amostras de áudio (>= 5 min)", file_count="multiple",
                            file_types=["audio"], type="filepath")
            epochs = gr.Slider(20, 300, value=100, step=10, label="Épocas")
            train_btn = gr.Button("Treinar", variant="primary")
            train_out = gr.Textbox(label="Status")
            train_btn.click(do_train, [name, files, epochs], train_out)

        with gr.Tab("2. Sintetizar"):
            song = gr.Dropdown(list_songs(), label="Música (partitura)")
            voice = gr.Dropdown(list_voices(), label="Voz treinada")
            refresh = gr.Button("↻ Atualizar listas")
            instrumental = gr.Audio(label="Instrumental p/ mix (opcional)", type="filepath")
            transpose = gr.Slider(-12, 12, value=0, step=1, label="Transpor (semitons)")
            synth_btn = gr.Button("Cantar 🎶", variant="primary")
            audio_out = gr.Audio(label="Resultado")
            synth_status = gr.Textbox(label="Status")

            refresh.click(lambda: (gr.update(choices=list_songs()),
                                   gr.update(choices=list_voices())),
                          outputs=[song, voice])
            synth_btn.click(do_synthesize, [song, voice, instrumental, transpose],
                            [audio_out, synth_status])

    return demo


if __name__ == "__main__":
    build_ui().launch()
