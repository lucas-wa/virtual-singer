"""Detecção de hardware e ajuste de parâmetros em runtime.

Como o usuário pode não saber o modelo da GPU, detectamos a VRAM disponível e
escolhemos batch size / fp16 / fatiamento de áudio automaticamente.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeProfile:
    device: str          # "cuda" | "cpu"
    gpu_name: str        # nome da GPU ou "CPU"
    vram_gb: float       # VRAM total em GB (0.0 se CPU)
    batch_size: int      # batch sugerido para inferência/treino
    use_fp16: bool       # usar meia-precisão
    max_segment_s: float # tamanho máximo de segmento de áudio processado de uma vez

    def summary(self) -> str:
        if self.device == "cpu":
            return "CPU (sem GPU CUDA detectada) — síntese será lenta."
        return (
            f"{self.gpu_name} | {self.vram_gb:.1f} GB VRAM | "
            f"batch={self.batch_size} | fp16={self.use_fp16} | "
            f"segmento<= {self.max_segment_s:.0f}s"
        )


def detect_profile() -> RuntimeProfile:
    """Inspeciona a GPU disponível e devolve um perfil de execução seguro."""
    try:
        import torch
    except ImportError:
        return RuntimeProfile("cpu", "CPU", 0.0, 1, False, 8.0)

    if not torch.cuda.is_available():
        return RuntimeProfile("cpu", "CPU", 0.0, 1, False, 8.0)

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024**3)
    name = props.name

    # Escolhe parâmetros por faixa de VRAM (conservador para deixar margem).
    if vram_gb >= 22:
        batch, fp16, seg = 16, True, 40.0
    elif vram_gb >= 11:
        batch, fp16, seg = 8, True, 30.0
    elif vram_gb >= 7:
        batch, fp16, seg = 4, True, 20.0
    else:
        batch, fp16, seg = 1, True, 12.0

    return RuntimeProfile("cuda", name, vram_gb, batch, fp16, seg)


if __name__ == "__main__":
    p = detect_profile()
    print(p.summary())
