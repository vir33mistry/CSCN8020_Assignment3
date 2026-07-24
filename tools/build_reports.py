"""Build the nine-page technical report and one-page Brightspace PDF."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "report"
SUBMISSION_DIR = PROJECT_ROOT / "submission"
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"

NAVY = colors.HexColor("#0F2742")
BLUE = colors.HexColor("#176B87")
LIGHT_BLUE = colors.HexColor("#DCEFF5")
ORANGE = colors.HexColor("#D97706")
GREEN = colors.HexColor("#16865A")
SLATE = colors.HexColor("#475569")
LIGHT = colors.HexColor("#F3F6F8")
WHITE = colors.white


def font_name(*, bold: bool = False) -> str:
    preferred = "ReportSans-Bold" if bold else "ReportSans"
    fallback = "Helvetica-Bold" if bold else "Helvetica"
    return (
        preferred
        if preferred in pdfmetrics.getRegisteredFontNames()
        else fallback
    )


def register_fonts() -> None:
    font_dir = Path("/usr/share/fonts/truetype/dejavu")
    regular = font_dir / "DejaVuSans.ttf"
    bold = font_dir / "DejaVuSans-Bold.ttf"
    if regular.is_file() and bold.is_file():
        pdfmetrics.registerFont(
            TTFont("ReportSans", str(regular))
        )
        pdfmetrics.registerFont(
            TTFont("ReportSans-Bold", str(bold))
        )


def make_styles() -> dict[str, ParagraphStyle]:
    samples = getSampleStyleSheet()
    base_font = font_name()
    bold_font = font_name(bold=True)

    return {
        "title": ParagraphStyle(
            "Title",
            parent=samples["Title"],
            fontName=bold_font,
            fontSize=24,
            leading=29,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=16,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=samples["Normal"],
            fontName=base_font,
            fontSize=12,
            leading=17,
            textColor=SLATE,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=samples["Heading1"],
            fontName=bold_font,
            fontSize=17,
            leading=21,
            textColor=NAVY,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=samples["Heading2"],
            fontName=bold_font,
            fontSize=12,
            leading=15,
            textColor=BLUE,
            spaceBefore=6,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=samples["BodyText"],
            fontName=base_font,
            fontSize=8.7,
            leading=12.2,
            textColor=colors.HexColor("#1E293B"),
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=samples["BodyText"],
            fontName=base_font,
            fontSize=7.2,
            leading=9.5,
            textColor=SLATE,
            spaceAfter=4,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=samples["BodyText"],
            fontName=bold_font,
            fontSize=10,
            leading=14,
            textColor=NAVY,
            backColor=LIGHT_BLUE,
            borderColor=BLUE,
            borderWidth=1,
            borderPadding=9,
            spaceBefore=6,
            spaceAfter=8,
        ),
        "equation": ParagraphStyle(
            "Equation",
            parent=samples["BodyText"],
            fontName=bold_font,
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            textColor=NAVY,
            backColor=LIGHT,
            borderPadding=8,
            spaceBefore=6,
            spaceAfter=8,
        ),
    }


def header_footer(
    canvas,
    document,
) -> None:
    canvas.saveState()
    width, height = letter
    page_number = canvas.getPageNumber()

    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 0.34 * inch, width, 0.34 * inch, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont(font_name(bold=True), 7.5)
    canvas.drawString(
        0.55 * inch,
        height - 0.225 * inch,
        "CSCN8020 Assignment 3 | Unitree G1 DQN",
    )

    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(
        0.55 * inch,
        0.42 * inch,
        width - 0.55 * inch,
        0.42 * inch,
    )
    canvas.setFillColor(SLATE)
    canvas.setFont(font_name(), 7.5)
    canvas.drawString(
        0.55 * inch,
        0.25 * inch,
        "Viraj Dipakkumar Mistry | 9088985",
    )
    canvas.drawRightString(
        width - 0.55 * inch,
        0.25 * inch,
        f"Page {page_number}",
    )
    canvas.restoreState()


def make_table(
    data: list[list[object]],
    *,
    column_widths: list[float] | None = None,
    font_size: float = 7.5,
    header_background=BLUE,
) -> Table:
    table = Table(
        data,
        colWidths=column_widths,
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), header_background),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("FONTNAME", (0, 0), (-1, 0), font_name(bold=True)),
                ("FONTNAME", (0, 1), (-1, -1), font_name()),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LEADING", (0, 0), (-1, -1), font_size + 2),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8C5CE")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def fitted_image(
    path: Path,
    *,
    max_width: float,
    max_height: float,
) -> Image:
    image = Image(str(path))
    ratio = min(
        max_width / image.imageWidth,
        max_height / image.imageHeight,
    )
    image.drawWidth = image.imageWidth * ratio
    image.drawHeight = image.imageHeight * ratio
    image.hAlign = "CENTER"
    return image


def bullet(text: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    return Paragraph(
        text,
        styles["body"],
        bulletText="•",
    )


def load_evidence() -> dict[str, object]:
    config_a = json.loads(
        (
            RESULTS_DIR
            / "config_a"
            / "training_summary.json"
        ).read_text(encoding="utf-8")
    )
    config_b = json.loads(
        (
            RESULTS_DIR
            / "config_b"
            / "training_summary.json"
        ).read_text(encoding="utf-8")
    )
    selection = json.loads(
        (
            RESULTS_DIR
            / "evaluation"
            / "selection_summary.json"
        ).read_text(encoding="utf-8")
    )
    selected_by_goal = pd.read_csv(
        RESULTS_DIR
        / "evaluation"
        / "config_a_by_goal.csv"
    )
    comparison = pd.read_csv(
        RESULTS_DIR
        / "evaluation"
        / "policy_comparison.csv"
    )
    return {
        "config_a": config_a,
        "config_b": config_b,
        "selection": selection,
        "selected_by_goal": selected_by_goal,
        "comparison": comparison,
    }


def build_technical_report() -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "DQN_Assignment_Report.pdf"
    styles = make_styles()
    evidence = load_evidence()
    config_a = evidence["config_a"]
    config_b = evidence["config_b"]
    selection = evidence["selection"]

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=0.58 * inch,
        rightMargin=0.58 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="Deep Q-Network Control of the Unitree G1 Left Elbow",
        author="Viraj Dipakkumar Mistry",
        subject="CSCN8020 Assignment 3",
    )
    story: list[object] = []

    # Page 1 - title and executive summary
    story.extend(
        [
            Spacer(1, 0.45 * inch),
            Paragraph(
                "Deep Q-Network Control of the Unitree G1 Left Elbow",
                styles["title"],
            ),
            Paragraph(
                "A Multi-Goal Reinforcement-Learning Study in MuJoCo and Gymnasium",
                styles["subtitle"],
            ),
            Spacer(1, 0.12 * inch),
            make_table(
                [
                    ["Student", "Viraj Dipakkumar Mistry"],
                    ["Student ID", "9088985"],
                    ["Course", "CSCN8020 - Reinforcement Learning"],
                    ["Selected model", "Configuration A - epsilon decay 0.995"],
                    ["Repository", "github.com/vir33mistry/CSCN8020_Assignment3"],
                ],
                column_widths=[1.45 * inch, 5.2 * inch],
                font_size=8.5,
                header_background=NAVY,
            ),
            Spacer(1, 0.22 * inch),
            Paragraph("Executive summary", styles["h1"]),
            Paragraph(
                (
                    "This project extends the completed Unitree MuJoCo G1 Primer "
                    "with a student-written PyTorch DQN. The approved observation, "
                    "actions, controller, reward, and success logic remain unchanged. "
                    "Two controlled 600-episode CPU experiments differ only in "
                    "epsilon decay. Both achieve 20/20 greedy successes across four "
                    "benchmark goals. Configuration A is selected because it reaches "
                    "a higher mean evaluation reward (13.1497) and lower mean final "
                    "absolute error (0.0071 rad) than Configuration B."
                ),
                styles["body"],
            ),
            Paragraph(
                (
                    "<b>Key result:</b> the selected DQN achieves 100% success, "
                    "exceeding the assignment's 80% threshold by 20 percentage points."
                ),
                styles["callout"],
            ),
            make_table(
                [
                    [
                        "Policy",
                        "Success",
                        "Mean reward",
                        "Mean length",
                        "Mean final |error|",
                    ],
                    ["Config A", "20/20", "13.1497", "20.75", "0.0071 rad"],
                    ["Config B", "20/20", "13.0880", "19.75", "0.0136 rad"],
                    ["Rule-based", "20/20", "12.8666", "24.00", "0.0122 rad"],
                ],
                column_widths=[
                    1.2 * inch,
                    0.85 * inch,
                    1.2 * inch,
                    1.1 * inch,
                    1.45 * inch,
                ],
                font_size=8,
            ),
            Spacer(1, 0.12 * inch),
            Paragraph(
                (
                    "Final validated environment: Python 3.12.13, Linux x86_64, "
                    "PyTorch 2.7.1 CPU, Gymnasium 1.3.0, MuJoCo 3.10.0."
                ),
                styles["small"],
            ),
            PageBreak(),
        ]
    )

    # Page 2 - environment
    story.extend(
        [
            Paragraph("1. Environment and baseline verification", styles["h1"]),
            Paragraph(
                (
                    "The primer provides a fixed-base G1 model, verified joint and "
                    "actuator mappings, proportional-derivative control, MuJoCo "
                    "bias-force compensation, and a compliant Gymnasium environment. "
                    "The DQN replaces only the high-level hand-written policy."
                ),
                styles["body"],
            ),
            make_table(
                [
                    ["Element", "Approved definition"],
                    [
                        "Observation",
                        "[elbow angle, angular velocity, goal, goal - angle]",
                    ],
                    ["Action 0", "Decrease the internal controller target by 0.08 rad"],
                    ["Action 1", "Hold the internal controller target"],
                    ["Action 2", "Increase the internal controller target by 0.08 rad"],
                    [
                        "Low-level control",
                        "PD torque plus qfrc_bias compensation and torque clipping",
                    ],
                    [
                        "Reward",
                        "-|error|; +1 in tolerance; -0.05 for non-HOLD near goal; +10 on success",
                    ],
                    [
                        "Success",
                        "|error| <= 0.04 rad for 8 consecutive environment steps",
                    ],
                    ["Time limit", "150 environment steps"],
                ],
                column_widths=[1.45 * inch, 5.3 * inch],
                font_size=7.6,
            ),
            Spacer(1, 0.12 * inch),
            Paragraph("Termination and truncation", styles["h2"]),
            Paragraph(
                (
                    "<b>terminated=True</b> means the task-defined success condition "
                    "was reached. <b>truncated=True</b> means the 150-step limit "
                    "ended the episode first. Both signals stop interaction. Replay "
                    "stores them separately, and only true termination masks the "
                    "Bellman bootstrap."
                ),
                styles["body"],
            ),
            Paragraph("Pre-training baseline evidence", styles["h2"]),
            make_table(
                [
                    ["Goal", "Episodes", "Success", "Mean reward", "Mean length", "Final |error|"],
                    ["-0.8", "5", "5/5", "10.5215", "27", "0.0123"],
                    ["-0.4", "5", "5/5", "15.2141", "21", "0.0132"],
                    ["+0.4", "5", "5/5", "15.2111", "21", "0.0127"],
                    ["+0.8", "5", "5/5", "10.5199", "27", "0.0106"],
                    ["Overall", "20", "20/20", "12.8666", "24.0", "0.0122"],
                ],
                column_widths=[
                    0.75 * inch,
                    0.8 * inch,
                    0.8 * inch,
                    1.05 * inch,
                    1.0 * inch,
                    1.1 * inch,
                ],
                font_size=7.5,
            ),
            Spacer(1, 0.12 * inch),
            bullet("Gymnasium environment checker passed.", styles),
            bullet("Rule-based validation succeeded in every benchmark episode.", styles),
            bullet("The fixed-base model and approved controller were not redesigned.", styles),
            PageBreak(),
        ]
    )

    # Page 3 - architecture
    story.extend(
        [
            Paragraph("2. Student-written DQN methodology", styles["h1"]),
            Paragraph("Q-network architecture", styles["h2"]),
            make_table(
                [
                    ["Layer", "Dimensions", "Activation", "Purpose"],
                    ["Input", "4", "-", "Approved observation vector"],
                    ["Hidden 1", "4 -> 64", "ReLU", "Non-linear state features"],
                    ["Hidden 2", "64 -> 64", "ReLU", "Higher-level interactions"],
                    ["Output", "64 -> 3", "None", "Unconstrained action Q-values"],
                ],
                column_widths=[
                    1.0 * inch,
                    1.15 * inch,
                    1.0 * inch,
                    3.55 * inch,
                ],
                font_size=7.8,
            ),
            Spacer(1, 0.12 * inch),
            Paragraph(
                (
                    "The final layer deliberately has no softmax. DQN needs relative, "
                    "unbounded estimates of discounted return rather than action "
                    "probabilities. ReLU layers use Kaiming initialization; the final "
                    "layer uses a small uniform initialization."
                ),
                styles["body"],
            ),
            Paragraph("Replay and target network", styles["h2"]),
            bullet(
                "ReplayBuffer has bounded capacity 50,000 and seeded uniform sampling without replacement.",
                styles,
            ),
            bullet(
                "Each transition stores state, action, reward, next state, terminated, and truncated.",
                styles,
            ),
            bullet(
                "Learning begins only after 500 transitions and uses mini-batches of 64.",
                styles,
            ),
            bullet(
                "The target network starts from the online weights and synchronizes every 250 updates.",
                styles,
            ),
            Paragraph("Bellman target and optimization", styles["h2"]),
            Paragraph(
                "y = r + 0.95 x (1 - terminated) x max Q_target(next_state, next_action)",
                styles["equation"],
            ),
            Paragraph(
                (
                    "The online network supplies Q(state, selected action) through "
                    "gather. The target is computed under no_grad and is detached. "
                    "Smooth L1 (Huber) loss, Adam with learning rate 0.001, and "
                    "gradient-norm clipping at 10 are applied before the optimizer "
                    "step."
                ),
                styles["body"],
            ),
            Paragraph("Checkpoint support", styles["h2"]),
            Paragraph(
                (
                    "Each checkpoint records network weights, target weights, optimizer "
                    "state, hyperparameters, epsilon, optimization count, seed, device, "
                    "and experiment metadata. Loading defaults to CPU and reconstructs "
                    "the architecture before applying the saved state dictionary."
                ),
                styles["body"],
            ),
            PageBreak(),
        ]
    )

    # Page 4 - reproducibility
    story.extend(
        [
            Paragraph("3. Training workflow and reproducibility", styles["h1"]),
            make_table(
                [
                    ["Parameter", "Baseline value"],
                    ["Discount factor gamma", "0.95"],
                    ["Learning rate", "0.001"],
                    ["Mini-batch size", "64"],
                    ["Replay capacity", "50,000"],
                    ["Initial / minimum epsilon", "1.00 / 0.05"],
                    ["Target-network update", "Every 250 optimization steps"],
                    ["Warm-up", "500 transitions"],
                    ["Episode limit", "150 environment steps"],
                    ["Training goal range", "[-0.8, +0.8] rad"],
                    ["Network", "4 -> 64 ReLU -> 64 ReLU -> 3"],
                    ["Loss / optimizer", "Huber / Adam"],
                ],
                column_widths=[2.45 * inch, 4.3 * inch],
                font_size=7.5,
            ),
            Spacer(1, 0.1 * inch),
            Paragraph("Controlled experiment", styles["h2"]),
            Paragraph(
                (
                    "Configuration A uses epsilon decay 0.995; Configuration B "
                    "uses 0.985. Both use 600 episodes, seed 8020, identical initial "
                    "weights, common goal-seed policy, CPU execution, and the same "
                    "hyperparameters. Only exploration decay changes."
                ),
                styles["body"],
            ),
            Paragraph("Seed controls", styles["h2"]),
            bullet("Python random, NumPy, and PyTorch are seeded with 8020.", styles),
            bullet("Replay sampling uses its own seeded NumPy generator.", styles),
            bullet("Each Gymnasium reset receives seed + episode index.", styles),
            bullet("Greedy evaluation uses the same fixed 20 seeds for every policy.", styles),
            Paragraph("Measured execution", styles["h2"]),
            Paragraph(
                (
                    f"Configuration A completed in {config_a['wall_clock_seconds']:.2f} "
                    f"seconds; Configuration B completed in "
                    f"{config_b['wall_clock_seconds']:.2f} seconds. The combined "
                    "required training time was about 1.56 minutes in the final "
                    "validated CPU run. A per-configuration safety timer is included."
                ),
                styles["callout"],
            ),
            Paragraph(
                (
                    "Training is headless. Rendering is isolated to saved-checkpoint "
                    "demonstration scripts. CUDA can be used only when explicitly "
                    "requested; CPU is the default and was the submitted validation path."
                ),
                styles["body"],
            ),
            PageBreak(),
        ]
    )

    # Page 5 - decay comparison
    story.extend(
        [
            Paragraph("4. Exploration-decay comparison", styles["h1"]),
            make_table(
                [
                    ["Required metric", "Config A: 0.995", "Config B: 0.985"],
                    ["Episodes", "600", "600"],
                    ["Wall-clock training", "53.70 s", "40.00 s"],
                    ["Final epsilon", "0.05", "0.05"],
                    ["Final-20 mean reward", "15.5566", "15.3677"],
                    ["Final-50 train success", "100%", "100%"],
                    ["Greedy evaluation", "20/20 (100%)", "20/20 (100%)"],
                    ["Mean evaluation reward", "13.1497", "13.0880"],
                    ["Mean final |error|", "0.0071 rad", "0.0136 rad"],
                    ["Mean episode length", "20.75", "19.75"],
                    ["HOLD fraction", "46.99%", "11.39%"],
                ],
                column_widths=[2.6 * inch, 2.05 * inch, 2.05 * inch],
                font_size=7.4,
            ),
            Spacer(1, 0.08 * inch),
            fitted_image(
                PLOTS_DIR / "epsilon_decay.png",
                max_width=7.0 * inch,
                max_height=2.5 * inch,
            ),
            Spacer(1, 0.06 * inch),
            fitted_image(
                PLOTS_DIR / "configuration_comparison.png",
                max_width=7.0 * inch,
                max_height=2.35 * inch,
            ),
            Paragraph(
                (
                    "Config B reaches minimum exploration near episode 200. Config A "
                    "explores longer, yet obtains the stronger reward and final-error "
                    "evidence after both policies reach 100% success."
                ),
                styles["small"],
            ),
            PageBreak(),
        ]
    )

    # Page 6 - curves
    story.extend(
        [
            Paragraph("5. Training curves and stability", styles["h1"]),
            fitted_image(
                PLOTS_DIR / "training_reward.png",
                max_width=7.0 * inch,
                max_height=3.6 * inch,
            ),
            Spacer(1, 0.06 * inch),
            fitted_image(
                PLOTS_DIR / "training_success_rate.png",
                max_width=7.0 * inch,
                max_height=2.5 * inch,
            ),
            Spacer(1, 0.06 * inch),
            fitted_image(
                PLOTS_DIR / "loss_curve.png",
                max_width=7.0 * inch,
                max_height=2.55 * inch,
            ),
            Paragraph(
                (
                    "Random early episodes produce large negative returns. Both "
                    "rolling success curves reach 100%. Huber loss is non-monotonic "
                    "because Q-target scale and replay composition change as terminal "
                    "bonuses become common; stable rewards and independent greedy "
                    "evaluation confirm that learning remains effective."
                ),
                styles["small"],
            ),
            PageBreak(),
        ]
    )

    # Page 7 - final evaluation
    selected_by_goal = evidence["selected_by_goal"]
    evaluation_rows = [
        ["Goal", "Episodes", "Successes", "Success rate", "Mean reward"]
    ]
    for row in selected_by_goal.itertuples():
        evaluation_rows.append(
            [
                f"{row.goal_angle:+.1f}",
                str(int(row.episodes)),
                str(int(row.successes)),
                f"{row.success_rate:.0%}",
                f"{row.mean_reward:.4f}",
            ]
        )
    evaluation_rows.append(
        ["Overall", "20", "20", "100%", "13.1497"]
    )
    story.extend(
        [
            Paragraph("6. Final greedy evaluation", styles["h1"]),
            Paragraph(
                (
                    "Evaluation disables exploration (epsilon 0.0) and runs five "
                    "episodes at each required goal. The selected checkpoint is loaded "
                    "rather than retrained."
                ),
                styles["body"],
            ),
            make_table(
                evaluation_rows,
                column_widths=[
                    1.0 * inch,
                    1.1 * inch,
                    1.1 * inch,
                    1.25 * inch,
                    1.35 * inch,
                ],
                font_size=8,
            ),
            Spacer(1, 0.13 * inch),
            fitted_image(
                PLOTS_DIR / "evaluation_success_by_goal.png",
                max_width=7.0 * inch,
                max_height=3.5 * inch,
            ),
            Paragraph(
                (
                    "<b>Threshold result:</b> 20/20 successes = 100%. The required "
                    "threshold is at least 16/20 = 80%. Configuration A exceeds the "
                    "threshold and succeeds at every positive and negative target."
                ),
                styles["callout"],
            ),
            Paragraph(
                (
                    "Per-goal mean final errors are 0.0093, 0.0050, 0.0102, and "
                    "0.0039 rad for -0.8, -0.4, +0.4, and +0.8 rad respectively. "
                    "This supports multi-goal generalization rather than success at "
                    "only one direction or magnitude."
                ),
                styles["body"],
            ),
            PageBreak(),
        ]
    )

    # Page 8 - baseline comparison
    story.extend(
        [
            Paragraph("7. Rule-based baseline versus selected DQN", styles["h1"]),
            make_table(
                [
                    ["Metric", "Rule-based", "Selected DQN"],
                    ["Successes / 20", "20", "20"],
                    ["Success rate", "100%", "100%"],
                    ["Mean cumulative reward", "12.8666", "13.1497"],
                    ["Mean episode length", "24.00", "20.75"],
                    ["Mean final absolute error", "0.0122 rad", "0.0071 rad"],
                    ["Mean action changes", "1.00", "6.75"],
                    ["HOLD action fraction", "68.75%", "46.99%"],
                    [
                        "Main behaviour",
                        "Direct move, then stable HOLD",
                        "Learned corrective actions; faster finish",
                    ],
                ],
                column_widths=[2.25 * inch, 2.15 * inch, 2.35 * inch],
                font_size=7.5,
            ),
            Spacer(1, 0.1 * inch),
            fitted_image(
                PLOTS_DIR / "rule_based_vs_selected_dqn.png",
                max_width=7.0 * inch,
                max_height=3.7 * inch,
            ),
            Paragraph(
                (
                    "The rule-based policy is most sample efficient because it needs "
                    "no learning, and it is more stable in action space. The DQN "
                    "generalizes across all goals, finishes sooner, earns higher "
                    "reward, and ends closer to the target, but changes action more "
                    "often. It uses HOLD appropriately, though less consistently than "
                    "the transparent baseline. A hand-written controller can excel in "
                    "this simple task because it begins with perfect structural "
                    "knowledge; DQN must infer that structure from finite experience."
                ),
                styles["body"],
            ),
            PageBreak(),
        ]
    )

    # Page 9 - recommendation, limitations, references
    story.extend(
        [
            Paragraph("8. Recommendation and conclusions", styles["h1"]),
            Paragraph(
                (
                    "<b>Recommendation:</b> select Configuration A. Success rate is "
                    "tied, so the decision uses multiple indicators: A has the higher "
                    "final-20 training reward, higher evaluation reward, lower final "
                    "error, and much greater HOLD use. Its additional measured training "
                    "time is negligible relative to the five-hour limit."
                ),
                styles["callout"],
            ),
            Paragraph("Limitations", styles["h2"]),
            bullet("Fixed-base, deterministic simulation controls one joint only.", styles),
            bullet("The primary comparison uses one training seed and 20 deterministic evaluation episodes.", styles),
            bullet("No disturbances, sensor noise, model mismatch, or physical-hardware effects are tested.", styles),
            bullet("The DQN changes actions more often than the rule-based policy.", styles),
            Paragraph("Future improvements", styles["h2"]),
            bullet("Repeat each configuration over several independent seeds.", styles),
            bullet("Evaluate a denser grid of unseen goals and controlled perturbations.", styles),
            bullet("With instructor approval, study action-switch regularization, Double DQN, or prioritized replay.", styles),
            bullet("Extend from one fixed-base joint to multi-joint and sim-to-real control.", styles),
            Paragraph("Submission reproducibility", styles["h2"]),
            Paragraph(
                (
                    "The repository includes explicit source, tests, metrics, plots, "
                    "both experiment checkpoints, selected checkpoint, completed "
                    "notebook, nine-page report, one-page submission PDF, exact "
                    "commands, and a 2-minute 15-second saved-policy video. The "
                    "interactive viewer script loads the checkpoint and uses epsilon "
                    "0.0 without retraining."
                ),
                styles["body"],
            ),
            Paragraph("AI-use acknowledgement", styles["h2"]),
            Paragraph(
                (
                    "Generative AI assistance supported scaffolding, debugging, test "
                    "design, formatting, and draft writing. The student remains "
                    "responsible for verification, understanding, and explanation."
                ),
                styles["body"],
            ),
            Paragraph("References", styles["h2"]),
            Paragraph(
                (
                    "1. Mnih et al. (2015), Human-level control through deep "
                    "reinforcement learning, Nature 518, 529-533, "
                    "doi:10.1038/nature14236.<br/>"
                    "2. PyTorch, Reinforcement Learning (DQN) Tutorial, "
                    "docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html.<br/>"
                    "3. Farama Foundation, Handling Time Limits, "
                    "gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/.<br/>"
                    "4. Google DeepMind, MuJoCo Python Documentation, "
                    "mujoco.readthedocs.io/en/stable/python.html.<br/>"
                    "5. CSCN8020 Assignment 3 specification and G1 Primer Workshop."
                ),
                styles["small"],
            ),
        ]
    )

    document.build(
        story,
        onFirstPage=header_footer,
        onLaterPages=header_footer,
    )
    return output_path


def wrap_text(
    canvas,
    text: str,
    *,
    x: float,
    y: float,
    width: float,
    font_name: str,
    font_size: float,
    leading: float,
) -> float:
    paragraph = Paragraph(
        text,
        ParagraphStyle(
            "CanvasParagraph",
            fontName=font_name,
            fontSize=font_size,
            leading=leading,
            textColor=colors.HexColor("#1E293B"),
        ),
    )
    _, height = paragraph.wrap(width, 4 * inch)
    paragraph.drawOn(canvas, x, y - height)
    return y - height


def build_brightspace_pdf() -> Path:
    from reportlab.pdfgen import canvas

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    output_path = (
        SUBMISSION_DIR
        / "CSCN8020_Assignment3_Brightspace.pdf"
    )
    page = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter

    page.setFillColor(NAVY)
    page.rect(0, height - 1.45 * inch, width, 1.45 * inch, fill=1, stroke=0)
    page.setFillColor(WHITE)
    page.setFont(font_name(bold=True), 23)
    page.drawCentredString(
        width / 2,
        height - 0.68 * inch,
        "CSCN8020 Assignment 3",
    )
    page.setFont(font_name(), 12)
    page.drawCentredString(
        width / 2,
        height - 1.02 * inch,
        "Deep Q-Network Control of the Unitree G1 Left Elbow",
    )

    y = height - 1.9 * inch
    page.setFillColor(NAVY)
    page.setFont(font_name(bold=True), 13)
    page.drawString(0.75 * inch, y, "Student information")
    y -= 0.3 * inch

    info = [
        ("Full name", "Viraj Dipakkumar Mistry"),
        ("Student ID", "9088985"),
        ("Course", "CSCN8020 - Reinforcement Learning"),
    ]
    for label, value in info:
        page.setFillColor(SLATE)
        page.setFont(font_name(bold=True), 9)
        page.drawString(0.82 * inch, y, label.upper())
        page.setFillColor(colors.HexColor("#0F172A"))
        page.setFont(font_name(), 11)
        page.drawString(2.1 * inch, y, value)
        y -= 0.29 * inch

    y -= 0.12 * inch
    page.setFillColor(NAVY)
    page.setFont(font_name(bold=True), 13)
    page.drawString(0.75 * inch, y, "Project summary")
    y -= 0.2 * inch

    summary = (
        "This project implements a student-written PyTorch Deep Q-Network to "
        "control the Unitree G1 left elbow in MuJoCo. The agent maps four "
        "observation values to three discrete actions and learns with replay "
        "memory, epsilon-greedy exploration, Bellman updates, and a target "
        "network. Two controlled exploration-decay experiments were trained "
        "headlessly on CPU and evaluated at four benchmark angles. Both achieved "
        "20/20 greedy successes. Configuration A was selected for its higher "
        "mean reward and lower final error. The repository contains reproducible "
        "source code, checkpoints, structured metrics, plots, tests, a completed "
        "notebook, technical report, and a saved-policy video demonstrating "
        "multiple goals without retraining."
    )
    y = wrap_text(
        page,
        summary,
        x=0.82 * inch,
        y=y,
        width=6.85 * inch,
        font_name=font_name(),
        font_size=10,
        leading=15,
    )

    y -= 0.28 * inch
    page.setFillColor(NAVY)
    page.setFont(font_name(bold=True), 13)
    page.drawString(0.75 * inch, y, "Repository links")
    y -= 0.31 * inch

    links = [
        (
            "GitHub repository",
            "https://github.com/vir33mistry/CSCN8020_Assignment3",
        ),
        (
            "Cloneable URL",
            "https://github.com/vir33mistry/CSCN8020_Assignment3.git",
        ),
    ]
    for label, url in links:
        page.setFillColor(SLATE)
        page.setFont(font_name(bold=True), 8.5)
        page.drawString(0.82 * inch, y, label.upper())
        page.setFillColor(BLUE)
        page.setFont(font_name(), 9.5)
        page.drawString(0.82 * inch, y - 0.21 * inch, url)
        page.linkURL(
            url,
            (
                0.82 * inch,
                y - 0.25 * inch,
                7.5 * inch,
                y + 0.04 * inch,
            ),
            relative=0,
        )
        y -= 0.58 * inch

    page.setFillColor(LIGHT_BLUE)
    page.roundRect(
        0.75 * inch,
        0.75 * inch,
        7.0 * inch,
        0.78 * inch,
        10,
        fill=1,
        stroke=0,
    )
    page.setFillColor(NAVY)
    page.setFont(font_name(bold=True), 11)
    page.drawString(
        0.98 * inch,
        1.18 * inch,
        "Selected result: 20/20 greedy successes (100%)",
    )
    page.setFont(font_name(), 9)
    page.drawString(
        0.98 * inch,
        0.94 * inch,
        "Configuration A | Mean reward 13.1497 | Mean final error 0.0071 rad",
    )

    page.setFillColor(SLATE)
    page.setFont(font_name(), 7.5)
    page.drawCentredString(
        width / 2,
        0.42 * inch,
        "Conestoga College | CSCN8020 Reinforcement Learning",
    )
    page.showPage()
    page.save()
    return output_path


def main() -> None:
    register_fonts()
    technical_report = build_technical_report()
    brightspace_pdf = build_brightspace_pdf()
    print(f"Created: {technical_report}")
    print(f"Created: {brightspace_pdf}")


if __name__ == "__main__":
    main()
