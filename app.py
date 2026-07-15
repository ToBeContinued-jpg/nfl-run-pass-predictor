
import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="NFL Run or Pass Predictor",
    page_icon="🏈",
    layout="centered"
)


@st.cache_resource
def load_model():
    """Load the saved preprocessing and classification pipeline."""
    return joblib.load("football_model.joblib")


def get_distance_group(yards_to_go):
    """Create the same distance group used during training."""
    if yards_to_go <= 3:
        return "1-3 yards"
    elif yards_to_go <= 6:
        return "4-6 yards"
    elif yards_to_go <= 10:
        return "7-10 yards"
    else:
        return "11+ yards"


model = load_model()


st.title("🏈 NFL Run or Pass Predictor")

st.write(
    "Enter the pre-snap game situation to estimate whether "
    "the offense is more likely to call a run or pass."
)

st.caption(
    "The model was trained using NFL play-by-play data from "
    "2021–2023 and evaluated on the 2024 season."
)


with st.form("prediction_form"):

    first_column, second_column = st.columns(2)

    with first_column:
        down = st.selectbox(
            "Down",
            options=[1, 2, 3, 4]
        )

        yards_to_go = st.number_input(
            "Yards to first down",
            min_value=1,
            max_value=99,
            value=10,
            step=1
        )

        quarter = st.selectbox(
            "Quarter",
            options=[1, 2, 3, 4]
        )

        score_differential = st.number_input(
            "Offense score differential",
            min_value=-60,
            max_value=60,
            value=0,
            step=1,
            help=(
                "Offense score minus opponent score. "
                "For example, enter -7 if the offense is losing by 7."
            )
        )

    with second_column:
        minutes_remaining = st.number_input(
            "Minutes remaining in quarter",
            min_value=0,
            max_value=15,
            value=10,
            step=1
        )

        seconds_remaining = st.number_input(
            "Additional seconds",
            min_value=0,
            max_value=59,
            value=0,
            step=1
        )

        yardline_100 = st.number_input(
            "Yards from opponent's end zone",
            min_value=1,
            max_value=99,
            value=75,
            step=1,
            help=(
                "A team at its own 25-yard line is 75 yards "
                "from the opponent's end zone."
            )
        )

        offense_timeouts = st.selectbox(
            "Offense timeouts remaining",
            options=[0, 1, 2, 3],
            index=3
        )

        defense_timeouts = st.selectbox(
            "Defense timeouts remaining",
            options=[0, 1, 2, 3],
            index=3
        )

    submitted = st.form_submit_button(
        "Predict play",
        use_container_width=True
    )


if submitted:

    quarter_seconds_remaining = (
        minutes_remaining * 60
        + seconds_remaining
    )

    distance_group = get_distance_group(yards_to_go)

    down_distance = (
        f"{down}_{distance_group}"
    )

    input_data = pd.DataFrame({
        "down": [down],
        "ydstogo": [yards_to_go],
        "qtr": [quarter],
        "quarter_seconds_remaining": [
            quarter_seconds_remaining
        ],
        "down_distance": [down_distance],
        "score_differential": [
            score_differential
        ],
        "yardline_100": [yardline_100],
        "posteam_timeouts_remaining": [
            offense_timeouts
        ],
        "defteam_timeouts_remaining": [
            defense_timeouts
        ]
    })

    pass_probability = (
        model.predict_proba(input_data)[0, 1]
    )

    run_probability = 1 - pass_probability

    predicted_play = (
        "PASS"
        if pass_probability >= 0.50
        else "RUN"
    )

    st.divider()

    st.subheader(f"Predicted play: {predicted_play}")

    pass_column, run_column = st.columns(2)

    with pass_column:
        st.metric(
            "Pass probability",
            f"{pass_probability:.1%}"
        )

    with run_column:
        st.metric(
            "Run probability",
            f"{run_probability:.1%}"
        )

    st.progress(float(pass_probability))

    st.caption(
        "This prediction represents historical play-calling "
        "patterns, not certainty about an individual play."
    )


with st.expander("About the model"):
    st.write(
        """
        This application uses a logistic-regression classifier.
        Its inputs include down, distance, quarter, time remaining,
        score differential, field position, and remaining timeouts.

        The model achieved approximately 66.2% accuracy and 0.608
        log loss on held-out 2024 play-by-play data.
        """
    )
