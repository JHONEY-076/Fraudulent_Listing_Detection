from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as DrawingPath


def render_workflow():
    font_paths = [
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    font_path = next((candidate for candidate in font_paths if candidate.exists()), None)
    if font_path is None:
        raise RuntimeError("한글 렌더링을 위해 맑은 고딕 또는 Noto Sans CJK 폰트가 필요합니다.")
    font = FontProperties(fname=str(font_path))
    plt.rcParams["svg.fonttype"] = "path"
    figure, axes = plt.subplots(figsize=(14.5, 15.5))
    figure.patch.set_facecolor("white")
    axes.set(xlim=(0, 17.2), ylim=(0, 18.2))
    axes.axis("off")
    figure.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.015)

    def box(center, label, width=8.4, height=1.25, fill="#edf4fc", font_size=21):
        horizontal, vertical = center
        axes.add_patch(FancyBboxPatch(
            (horizontal - width / 2, vertical - height / 2), width, height,
            boxstyle="round,pad=0,rounding_size=0.16", linewidth=1.8,
            edgecolor="#426381", facecolor=fill, zorder=3,
        ))
        axes.text(horizontal, vertical, label, fontproperties=font, fontsize=font_size,
                  ha="center", va="center", color="#162b40", linespacing=1.6, zorder=4)

    def arrow(points):
        drawing_path = DrawingPath(
            points, [DrawingPath.MOVETO] + [DrawingPath.LINETO] * (len(points) - 1),
        )
        axes.add_patch(FancyArrowPatch(
            path=drawing_path, arrowstyle="-|>", mutation_scale=26,
            linewidth=2.8, color="#243b53", joinstyle="round", capstyle="round", zorder=5,
        ))

    axes.text(8.6, 17.65, "허위매물 분류 연구 과정", fontproperties=font, fontsize=28,
              fontweight="bold", ha="center", va="center", color="#162b40")
    axes.text(8.6, 17.05, "개발 데이터로 선택 → 보관 데이터로 검증 → test 예측·제출",
              fontproperties=font, fontsize=17, ha="center", va="center", color="#526b82")

    box((6.0, 15.9), "train.csv · 2,452건\n정답 라벨 포함")
    box((6.0, 13.7), "개발 데이터 · 1,961건\n학습·모델 선택에 사용")
    box((13.25, 13.7), "holdout · 491건\n최종 검증용 보관",
        width=4.5, fill="#fff2d8", font_size=19)
    box((6.0, 11.5), "7개 모델군 · 25개 설정\n동일한 층화 5-fold로 비교")
    box((6.0, 9.3), "OOF 예측 수집\n각 행을 학습하지 않은 모델의 예측", font_size=20)
    box((6.0, 7.1), "Recall ≥ 80%에서 F1 최대화\nCatBoost · 임계값 0.5077 선택")
    box((6.0, 4.9), "선택 모델 최종 학습\n개발 데이터 1,961건 사용")
    box((6.0, 2.7), "최종 내부 검증\nRecall 86.67% · F1 85.95%", fill="#e5f4ed")
    box((13.25, 7.1), "test.csv · 613건\n정답 라벨 없음", width=4.5, font_size=19)
    box((13.25, 4.9), "고정 모델로 test 예측\n임계값 재조정 없음", width=4.5, font_size=18)
    box((13.25, 2.7), "submission.csv\n정상 551 · 허위 의심 62",
        width=4.5, fill="#e5f4ed", font_size=18)

    for start_height, end_height in [(15.9, 13.7), (13.7, 11.5), (11.5, 9.3),
                                     (9.3, 7.1), (7.1, 4.9), (4.9, 2.7)]:
        arrow([(6.0, start_height - 0.67), (6.0, end_height + 0.67)])
    arrow([(10.25, 15.9), (13.25, 15.9), (13.25, 14.37)])
    arrow([(15.55, 13.7), (16.25, 13.7), (16.25, 1.3), (6.0, 1.3), (6.0, 2.03)])
    arrow([(10.25, 4.9), (10.95, 4.9)])
    arrow([(13.25, 6.43), (13.25, 5.57)])
    arrow([(13.25, 4.23), (13.25, 3.37)])

    axes.text(16.7, 9.5, "보관 데이터 · 모델 선택에 사용하지 않음", rotation=270,
              fontproperties=font, fontsize=15, ha="center", va="center", color="#74521d")
    axes.text(8.6, 0.6, "모델 선택: 개발 OOF   |   성능 보고: holdout   |   대회 제출: test 예측",
              fontproperties=font, fontsize=15, ha="center", va="center", color="#526b82")

    output = Path(__file__).resolve().parent / "images"
    output.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "svg"]:
        destination = output / f"research_workflow.{extension}"
        figure.savefig(destination, dpi=160, facecolor="white")
        print(destination)
    plt.close(figure)


if __name__ == "__main__":
    render_workflow()
