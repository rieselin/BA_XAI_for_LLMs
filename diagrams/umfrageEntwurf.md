# Flowchart umfrage: 

``` mermaid 
flowchart TD
    A([Start Survey]) --> B

    subgraph BG["`Person Background`"]
        B[Age Group] --> C[Gender]
        C --> D[Job Title / Professional Group]
        D --> E[Industry]
    end

    E --> F{"`Do you know<br/>AI / LLMs?<br/>e.g. ChatGPT, Claude...`"}
    F -- No --> Z2([End Survey])
    F -- Yes --> G{"`Have you ever<br/>used an LLM?`"}

    G -- No --> H["`Why not?<br/>(Multiple Choice + Other)`"]
    H --> Z2

    G -- Yes --> I["`Which LLMs have you used?<br/>(Multiple selection possible)<br/>• ChatGPT<br/> • Claude<br/> • LLaMA<br/> • Other`"]

    I --> J["`How would you rate your knowledge<br/>about LLMs?<br/>1 = No idea, 5 = Expert`"]

    J --> K["`Which of these tools are<br/>a form of AI?<br/>(AI ≠ LLM <br/>Check awareness)`"]

    %% ✅ Quality Screenout
    K --> K_CHECK{"`Quality check:<br/>Selected analog clock?`"}
    K_CHECK -- Yes --> Z2
    K_CHECK -- No --> L

    subgraph RESEARCH["`Research Questions`"]
        L["`Evaluate LLM responses`"]

        subgraph TRUST1["`Trust`"]
            L --> L1["Evaluate Trust"]
        end

        L1 --> P["`Do you know XAI?<br/>(Explainable AI)`"]

        %% ✅ Randomized Assignment
        P --> RAND{"`Randomization<br/>XAI Groups`"}
        RAND -- "50%: all methods" --> Q1
        RAND -- "17%: only method 1" --> Q1
        RAND -- "17%: only method 2" --> Q2
        RAND -- "16%: only method 3" --> Q3

        %% Method 1
        Q1["`Show XAI Method COT`"]

        subgraph TRUST_XAI1["`Trust after Method COT`"]
            Q1 --> Q1A["`Evaluate Trust`"]
        end

        %% Method 2
        Q1A --> Q2["`Show XAI Method SHAP`"]

        subgraph TRUST_XAI2["`Trust after Method SHAP`"]
            Q2 --> Q2A["`Evaluate Trust`"]
        end

        %% Method 3
        Q2A --> Q3["`Show XAI Method 3`"]

        subgraph TRUST_XAI3["`Trust after Method CONF`"]
            Q3 --> Q3A["`Evaluate Trust`"]
        end

        %% If only one method → continue directly
        Q1A --> R
        Q2A --> R
        Q3A --> R

        R{"`Format of the<br/>Explanation<br/>(random)`"}
        R -- Video --> S["`Video about<br/>how LLMs work`"]
        R -- Text --> T["`Text explanation about<br/>how LLMs work`"]

        subgraph TRUST_FINAL["`Trust after Video/Text`"]
            S --> U1["`Evaluate Trust`"]
            T --> U1
        end

        U1 --> V["`Reflection<br/>Has your trust<br/>changed due to the explanations?<br/>(Yes / No / Unsure)<br/>→ if Yes: What changed?<br/>(open text field)`"]
    end

    V --> Z2([End Survey])

    style BG fill:#e8f4f8,stroke:#4a90d9,color:#000
    style RESEARCH fill:#f0f8e8,stroke:#5aaa35,color:#000
    style TRUST1 fill:#fff8e1,stroke:#f9a825,color:#000
    style TRUST_XAI1 fill:#fce4ec,stroke:#e91e63,color:#000
    style TRUST_XAI2 fill:#fce4ec,stroke:#e91e63,color:#000
    style TRUST_XAI3 fill:#fce4ec,stroke:#e91e63,color:#000
    style TRUST_FINAL fill:#fff3e0,stroke:#ef6c00,color:#000
    style Z2 fill:#ffdddd,stroke:#cc0000,color:#000
    style A fill:#ddffdd,stroke:#00aa00,color:#000

```