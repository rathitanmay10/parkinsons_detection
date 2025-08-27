graph TD
    A[Start: Get the raw voice data] --> B{Step 1: Prepare the Data};
    B --> C{Step 2: Create New Data <br> (The Forger vs. Expert Game)};
    C --> D{Step 3: Build the Full Team};
    D --> E{Step 4: Train the Final Judge};
    E --> F{Step 5: The Final Exam};
    F --> G[End: Get a Prediction!];

    subgraph " "
        style A fill:#D6EAF8,stroke:#333,stroke-width:2px
        style G fill:#D5F5E3,stroke:#333,stroke-width:2px
    end
    
    %% Descriptions
    click A "Includes 195 voice recordings with different measurements (jitter, shimmer, etc.)."
    click B "Clean up the data so the computer can understand it. This is like organizing your notes before an exam."
    click C "Use the 'Forger' AI to create brand new, realistic voice samples of Parkinson's patients to balance our team."
    click D "Combine the original training data with the new fake data. Now we have a big, balanced team of examples."
    click E "Train a different AI (the Random Forest 'Judge') on this big, balanced dataset. Its only job is to learn how to tell the difference between 'healthy' and 'Parkinson''s."
    click F "Give the 'Judge' brand new voice data it has never seen before and test how well it does."
    click G "The Judge makes its final call: This voice sample is likely from a healthy person (0) or someone with Parkinson's (1)."
    
