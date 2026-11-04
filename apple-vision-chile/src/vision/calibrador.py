"""Conversión px -> mm usando un objeto de referencia de tamaño conocido."""


def calcular_px_por_mm(ancho_referencia_px: float, ancho_referencia_mm: float) -> float:
    """Calcula el factor de calibración a partir de un objeto de medida real conocida.

    Medir el objeto una sola vez con la cámara fija y copiar el resultado a
    `px_por_mm` en config.yaml.
    """
    if ancho_referencia_mm <= 0:
        raise ValueError("El ancho de referencia en mm debe ser > 0")
    return ancho_referencia_px / ancho_referencia_mm


def px_a_mm(valor_px: float, px_por_mm: float) -> float:
    return valor_px / px_por_mm
