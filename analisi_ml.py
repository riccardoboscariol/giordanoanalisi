import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

st.title("🧠 Analisi Combinazioni Risposte Target/Controllo")

uploaded_file = st.file_uploader("Carica il file CSV generato", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.success("CSV caricato!")

    # Separazione dati
    test_df = df[df["type"] == "test"].copy()
    target_df = df[df["type"] == "target"][["participant_id", "response"]].rename(columns={"response": "resp_target"})
    control_df = df[df["type"] == "control"][["participant_id", "response"]].rename(columns={"response": "resp_control"})

    # Merge risposte target e controllo
    combos = target_df.merge(control_df, on="participant_id")
    
    # Crea etichetta combinata
    def combo_label(row):
        if row["resp_target"] == True and row["resp_control"] == False:
            return "Target TRUE / Control FALSE"
        elif row["resp_target"] == False and row["resp_control"] == True:
            return "Target FALSE / Control TRUE"
        elif row["resp_target"] == True and row["resp_control"] == True:
            return "Entrambe TRUE"
        else:
            return "Entrambe FALSE"

    combos["target_combo"] = combos.apply(combo_label, axis=1)

    # Costruzione features: risposte ai test (30)
    feature_rows = []
    for pid, group in test_df.groupby("participant_id"):
        ordered = group.sort_values("frase")  # Ordine garantito
        feature_rows.append({
            "participant_id": pid,
            **{f"q{i+1}": bool(r) for i, r in enumerate(ordered["response"].tolist())}
        })

    features_df = pd.DataFrame(feature_rows)

    # Merge con target_combo
    dataset = features_df.merge(combos[["participant_id", "target_combo"]], on="participant_id")

    st.write("Anteprima dataset pronto per l'analisi:")
    st.dataframe(dataset.head())

    # Addestramento modello
    X = dataset[[f"q{i+1}" for i in range(30)]]
    y = dataset["target_combo"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)

    st.subheader("📊 Report del modello")
    st.text(classification_report(y, y_pred))

    # Confusion matrix
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay.from_estimator(model, X, y, ax=ax, cmap="Blues", xticks_rotation=45)
    st.pyplot(fig)

    # Analisi delle classi più comuni
    st.subheader("🔍 Distribuzione delle classi (combinazioni target/control)")
    st.bar_chart(dataset["target_combo"].value_counts())

    st.info("Interpretazione: se la maggioranza dei compilatori con pattern coerente cade in una specifica combinazione, "
            "questa combinazione potrebbe essere più plausibile.")
