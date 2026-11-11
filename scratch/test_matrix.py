import matplotlib.mathtext as mathtext

parser = mathtext.MathTextParser('path')
try:
    parser.parse(r"$\left[ 1 \quad 2 \right]$")
    print("left[ ... right] works!")
except Exception as e:
    print("Failed:", e)

