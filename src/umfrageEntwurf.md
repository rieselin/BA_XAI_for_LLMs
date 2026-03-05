Flowchart umfrage: 

Person backgound: Altersgruppe u (Gender?) u Berufssektor u Berufsgruppe

Check: do you know what AI LLM is? (here add beispiele) => if no => end of umfrage
Check: have used llm before => if not why? (multiple choice + other) => end of umfrage
    if yes => which ones? (multiple choice + several answers possible)
        - claude
        - chat gpt
        - llama
        - ...
        - other + angabe in textfeld
how would you rate your knowledge on how llms work (1 no idea, 5 = i have extensive knowledge)

    other AI beispiele => ask to select which of these are a kind of AI (figure out if aware that ai != llm)


need to know for antwort einordnung:
    Person background
    AI / LLM knowledge


forschungsfragen: 
    show several outputs from an llm (answering alltagsfragen? answering krankheitsfragen?)
        also add obviously factually wrong ones
    ask user to rate how much they trust that answer is factually correct 
    - if possible: llm einbinden damit user direkt frage stellen kann + trust bewerten lassen

    ask about knowledge of xai

    show xai outputs and ask about trust again
    - show output of different XAI methods 

    show video? show text explanation ? on how llms work
    - ask about trust again
end umfrage


``` mermaid 
flowchart TD
    A([Start Umfrage]) --> B

    subgraph BG["`Person Background`"]
        B[Altersgruppe] --> C[Gender]
        C --> D[Berufssektor]
        D --> E[Berufsgruppe]
    end

    E --> F{"`Kennst du<br/>AI / LLMs?<br/>e.g. ChatGPT, Claude...`"}
    F -- Nein --> Z2([Ende Umfrage])
    F -- Ja --> G{"`Hast du schon mal<br/>ein LLM benutzt?`"}

    G -- Nein --> H["`Warum nicht?<br/>(Multiple Choice + Andere)`"]
    H --> Z2

    G -- Ja --> I["`Welche LLMs hast du benutzt?<br/>(Mehrfachauswahl möglich)<br/>• ChatGPT<br/> • Claude<br/> • LLaMA<br/> • Andere`"]

    I --> J["`Wie schätzt du dein Wissen<br/>über LLMs ein?<br/>1 = K.A, 5 = Experte`"]

    J --> K["`Welche dieser Tools sind<br/>eine Form von KI?<br/>(AI ≠ LLM <br/>Bewusstsein prüfen)`"]

    subgraph RESEARCH["`Forschungsfragen`"]
        K --> L["`LLM-Antworten bewerten<br/>(Alltags- & <br/>Gesundheitsfragen,<br/>auch faktisch falsche<br/>Antworten)`"]

        subgraph TRUST1["`Vertrauen — 3 Dimensionen`"]
            L --> L1["`① Behavioral Trust (1–5)<br/><i>Würde Antwort ohne Prüfung verwenden</i>`"]
            L1 --> L2["`② Perceived Accuracy (1–5)<br/><i>Halte Antwort für faktisch korrekt</i>`"]
            L2 --> L3["`③ Social Trust (1–5)<br/><i>Würde Antwort weiterempfehlen</i>`"]
        end

        L3 --> N{"`Live LLM<br/>verfügbar?`"}
        N -- Ja --> O["`Eigene Frage stellen`"]
        N -- Nein --> P

        subgraph TRUST2["`Vertrauen — 3 Dimensionen`"]
            O --> O1["`① Behavioral Trust (1–5)`"]
            O1 --> O2["`② Perceived Accuracy (1–5)`"]
            O2 --> O3["`③ Social Trust (1–5)`"]
        end

        O3 --> P["`Kennst du XAI?<br/>(Explainable AI)`"]

        P --> Q1["`XAI Methode 1 anzeigen`"]

        subgraph TRUST_XAI1["`Vertrauen nach Methode 1`"]
            Q1 --> Q1A["`① Behavioral Trust (1–5)`"]
            Q1A --> Q1B["`② Perceived Accuracy (1–5)`"]
            Q1B --> Q1C["`③ Social Trust (1–5)`"]
        end

        Q1C --> Q2["`XAI Methode 2 anzeigen`"]

        subgraph TRUST_XAI2["`Vertrauen nach Methode 2`"]
            Q2 --> Q2A["`① Behavioral Trust (1–5)`"]
            Q2A --> Q2B["`② Perceived Accuracy (1–5)`"]
            Q2B --> Q2C["`③ Social Trust (1–5)`"]
        end

        Q2C --> Q3["`XAI Methode 3 anzeigen`"]

        subgraph TRUST_XAI3["`Vertrauen nach Methode 3`"]
            Q3 --> Q3A["`① Behavioral Trust (1–5)`"]
            Q3A --> Q3B["`② Perceived Accuracy (1–5)`"]
            Q3B --> Q3C["`③ Social Trust (1–5)`"]
        end

        Q3C --> R{"`Format der<br/>Erklärung<br/>(zufällig)`"}
        R -- Video --> S["`Video über<br/>LLM-Funktionsweise`"]
        R -- Text --> T["`Text-Erklärung über<br/>LLM-Funktionsweise`"]

        subgraph TRUST_FINAL["`Vertrauen nach Video/Text — 3 Dimensionen`"]
            S --> U1["`① Behavioral Trust (1–5)`"]
            T --> U1
            U1 --> U2["`② Perceived Accuracy (1–5)`"]
            U2 --> U3["`③ Social Trust (1–5)`"]
        end

        U3 --> V["`Reflexion<br/>Hat sich dein Vertrauen <br/>durchdie Erklärungen<br/> verändert?<br/>(Ja / Nein / Unsicher)<br/>→ falls Ja/Nein: Was hat <br/>sich verändert?<br/>(offenes Textfeld)`"]
    end

    V --> Z2([Ende Umfrage])

    style BG fill:#e8f4f8,stroke:#4a90d9,color:#000
    style RESEARCH fill:#f0f8e8,stroke:#5aaa35,color:#000
    style TRUST1 fill:#fff8e1,stroke:#f9a825,color:#000
    style TRUST2 fill:#fff8e1,stroke:#f9a825,color:#000
    style TRUST_XAI1 fill:#fce4ec,stroke:#e91e63,color:#000
    style TRUST_XAI2 fill:#fce4ec,stroke:#e91e63,color:#000
    style TRUST_XAI3 fill:#fce4ec,stroke:#e91e63,color:#000
    style TRUST_FINAL fill:#fff3e0,stroke:#ef6c00,color:#000
    style Z2 fill:#ffdddd,stroke:#cc0000,color:#000
    style A fill:#ddffdd,stroke:#00aa00,color:#000

```