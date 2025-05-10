import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

st.title("📊 Analisi Risposte dei Compilatori Virtuali")

uploaded_file = st.file_uploader("Carica il file CSV simulato", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.success("File caricato correttamente.")
    st.write("Esempio di dati:", df.head(3))

    # Filtra e calcola accuratezza
    test_df = df[df["type"] == "test"].copy()
    test_df["correct"] = test_df["response"] == test_df["corretta"]

    acc_df = test_df.groupby("participant_id")["correct"].mean().reset_index()
    acc_df = acc_df.rename(columns={"correct": "accuracy_test"})

    # Estrai risposte target e controllo
    target_df = df[df["type"] == "target"][["participant_id", "response"]].rename(columns={"response": "target_response"})
    control_df = df[df["type"] == "control"][["participant_id", "response"]].rename(columns={"response": "control_response"})

    # Merge
    full_df = acc_df.merge(target_df, on="participant_id").merge(control_df, on="participant_id")

    st.subheader("Distribuzione dell'accuratezza nei test")
    fig, ax = plt.subplots()
    sns.histplot(full_df["accuracy_test"], bins=10, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Scegli la frase da analizzare")
    scelta = st.radio("Quale risposta vuoi prevedere?", ("Target", "Controllo"))

    y_col = "target_response" if scelta == "Target" else "control_response"
    y = full_df[y_col].astype(int)
    X = full_df[["accuracy_test"]]

    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=0),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0)
    }

    st.subheader("Risultati dei Modelli")

    for name, model in models.items():
        model.fit(X, y)
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)

        auc = roc_auc_score(y, y_prob)
        acc = accuracy_score(y, y_pred)

        st.markdown(f"**{name}**")
        st.write(f"AUC: {auc:.2f}, Accuracy: {acc:.2f}")

        fpr, tpr, _ = roc_curve(y, y_prob)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - {name}")
        ax.legend()
        st.pyplot(fig)