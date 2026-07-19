import streamlit as st  # Importing the Streamlit library for creating web applications
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import time
from pypdf import PdfReader
from pydantic import BaseModel, Field
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# hide deprication warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Pharma Assistant",
    initial_sidebar_state = 'auto',
    layout='wide'
)

# hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea 
hide_streamlit_style = """
	<style>
    #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
    #GithubIcon {visibility: hidden;}
    .styles_viewerBadge__1yB5_ {display: none;}
    .viewerBadge_link__1S137 {display: none;}
    .viewerBadge_text__1JaDK {display: none;
    div.block-container{padding-top:2rem;}
    div.stButton {text-align:center;}
    </style>
"""

# hide the CSS code from the screen as they are embedded in markdown text. 
# Also, allow streamlit to unsafely process as HTML
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.title("Pharma Assistant")

with st.sidebar:
    st.title("Navigation")
    selected = st.radio(
        "Choose an option:",
        ["Price Discovery", "Market Scoring - Africa", "Regulatory and Compliance"]
    )

if(selected == 'Market Scoring - Africa'):
    st.header('Market Scoring - Africa')
    st.write("Instructional Text for market scoring")

    def assign_decision(score):
        if score >= q1:
            return "🟢"
        elif score >= q3:
            return "🟡"
        else:
            return "🔴"

    def generate_llm_insights_from_csv(df_for_llm: dataframe):
        # 1. Minimize tokens by sending a summary + a small sample
        # (Instead of passing the entire file, we calculate key metrics locally)
        row_count = len(df_for_llm)
        columns = list(df_for_llm.columns)
        numeric_summary = df_for_llm.describe().to_dict()
        #sample_data = df_for_llm.head(5).to_string(index=False) 
        mod_df_for_llm = df_for_llm[['Country', 'Market_Score', 'Execution_Decision']]

        # 3. Create a compact, token-efficient prompt
        system_prompt = f"""
        You are a data analyst. I have a dataset with {row_count} total rows.
        Columns: {', '.join(columns)}

        Numeric Summary:
        {numeric_summary}

        Data with population, health spends and deaths removed:
        {mod_df_for_llm}

        Please answer the user prompt using this data only.
        """

        user_prompt = """
        For each of the country data loaded by the system prompt, do the following:
        TABLE DATA RULES
        1) Create a table having three columns for ALL the countries:
           - Country Name
           - Executive Decision
           - Market Scoring
           - Expansion recommendations
           - Risk of expansion
        2) For the Executive Decision column, copy the value directly from the original dataframe.   
        2) Write the expansion recommendations in a bulleted list with atleast 3 short sentences 
        3) Write the risk of expansion in a short 1 line sentence (max 10 words) 
        
        TABLE STYLING RULES
        1) Return the above in an html table format.
        2) The background color of the html table should be black but the text should be in white.
        3) Output ONLY valid HTML code. Do not include introductory text, explanations, or concluding remarks.
        4) Do not use markdown code blocks (like ```html).
        """

        # 4. Initialize Gemini Client and call the LLM
        llm = OpenAI(
            api_key=st.secrets["OPENAI_API_KEY"], base_url="https://api.deepseek.com"
        )

        response = llm.chat.completions.create(
            model="deepseek-chat",  # For general QA / deepseek-reasoning for R1
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,  # Low temperature forces truthfulness to the context
        )
        return response.choices[0].message.content
    
    try:
        if st.button("Load WHO database and Derive Strategic Execution Recommendations", type="primary"):
            # Define the local directory path containing your CSVs
            # Replace this with your actual folder path
            CSV_DIRECTORY = "./datasets/market_scoring"

            # Ensure the directory exists to avoid errors
            if not os.path.exists(CSV_DIRECTORY):
                st.error(f"The directory '{CSV_DIRECTORY}' does not exist. Please create it and add your CSVs.")
            else:
                # Filter and find up to 3 CSV files in the target directory
                all_files = os.listdir(CSV_DIRECTORY)
                csv_files = [f for f in all_files if f.lower().endswith('.csv')][:3]

                if len(csv_files) < 3:
                    st.warning(f"Found {len(csv_files)} CSV(s). Please place at least 3 CSVs in '{CSV_DIRECTORY}'.")
                
                # Iterate through the files and display loading status
                for file_name in csv_files:
                    # Create a placeholder row for each file to update dynamically
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**{file_name}**...")
                    with col2:
                        # Show an active loading spinner
                        with st.spinner(""):
                            full_path = os.path.join(CSV_DIRECTORY, file_name)
                            
                            # Read the actual CSV file using pandas
                            if '1_afro_population_test' in file_name:
                                df_pop = pd.read_csv(full_path, thousands=',')
                            if '2_total_deaths_by_country' in file_name:
                                df_death = pd.read_csv(full_path, thousands=',')
                            if '3_afr_ghed_chegdp_sha2022_test' in file_name:
                                df_exp = pd.read_csv(full_path, thousands=',')    

                            # Optional simulated delay to visualising loading for small files
                            time.sleep(1.0) 
                            col2.write("✅")
                st.success("All 3 CSV files loaded successfully!")            
    
                # Assuming common column is named 'Country'
                merged = df_pop.merge(df_death, on="Country").merge(df_exp, on="Country")
                
                # Fill missing values to prevent math errors

                # Put a mean to the rest of the GDP total
                merged["Health_Spend_(in_$_billion)"] = (merged["GDP_Total"] * (merged["Expenditure"] / 100))

                # Market Scoring Logic
                # Formula: Score = (Population_Weight * Normalized_Pop) + (Spend_Weight * Normalized_Spend) - (Death_Weight * Normalized_Death_Rate)
                # We assume higher spend & pop yield a better score, while higher death rates lower it.

                # Calculate percentiles (0 to 1 scale) to normalize
                pop_norm = merged['Population'] / merged['Population'].max()
                spend_norm = merged['Health_Spend_(in_$_billion)'] / merged['Health_Spend_(in_$_billion)'].max()
                death_norm = merged['Deaths'] / merged['Deaths'].max()

                # Assign weights (Total = 1.0)
                w_pop, w_spend, w_death = 0.4, 0.4, 0.2
            
                merged['Market_Score'] = (
                    (pop_norm * w_pop) + 
                    (spend_norm * w_spend) - 
                    (death_norm * w_death)
                )
                
                # Scale to 0-100 and sort
                merged['Market_Score'] = (merged['Market_Score'] - merged['Market_Score'].min()) / (merged['Market_Score'].max() - merged['Market_Score'].min()) * 100
                merged = merged.sort_values(by='Market_Score', ascending=False).reset_index(drop=True)

                # 4. Presentation & Execution Decision
                st.subheader("📊 Market Opportunity Ranking")
                
                # Display the legend first
                st.caption("Legend")
                st.markdown("🟢 Execute (High Priority) 🟡 Monitor (Moderate Potential) 🔴 Evaluate (Low Priority)")
                
                # Segment and display the decision matrix
                q1 = merged['Market_Score'].quantile(0.75)
                q3 = merged['Market_Score'].quantile(0.25)

                merged['Execution_Decision'] = merged['Market_Score'].apply(assign_decision)
                
                merged = merged.round({'Population': 0, 'Health_Spend_(in_$_billion)': 0, 'Deaths': 0, 'Market_Score': 0})
                merged_formatted = merged

                # Now change the data type of large numbers to text so that we can put the thousands seperator
                merged_formatted['Population'] = merged_formatted['Population'].map('{:,.0f}'.format)
                merged_formatted['Health_Spend_(in_$_billion)'] = merged_formatted['Health_Spend_(in_$_billion)'].map('{:,.0f}'.format)
                merged_formatted['Deaths'] = merged_formatted['Deaths'].map('{:,.0f}'.format)

                st.dataframe(merged_formatted[['Country', 'Population', 'Health_Spend_(in_$_billion)', 'Deaths', 'Market_Score', 'Execution_Decision']], use_container_width=True)
                
                # 5. Visual Summary
                st.subheader("📈 Visualization of Top 10 Markets")
                top_10 = merged.head(10)
                
                fig, ax = plt.subplots(figsize=(5, 2), facecolor='none')
                plt.rcParams.update({'font.size': 4})
                ax.set_facecolor('black')
                ax.set_xlabel('Categories', color='white')
                ax.set_ylabel('Values', color='white')
                ax.tick_params(colors='white', which='both')

                for spine in ax.spines.values():
                    spine.set_color('white')
                    spine.set_linewidth(0.3)

                bars = ax.barh(top_10['Country'][::-1], top_10['Market_Score'][::-1], color='cornflowerblue')
                ax.set_xlabel("Market Score (0-100)")
                ax.set_title("Top 10 African Markets by Execution Score", color='white')
                st.pyplot(fig)

                # 6. Actionable Summary
                st.subheader("🚀 Strategic Execution Recommendations")
                
                with st.spinner("Deriving recommendation..."):
                    # Add your specific logic/code here
                    ai_output = generate_llm_insights_from_csv(merged[['Country', 'Population', 'Health_Spend_(in_$_billion)', 'Deaths', 'Market_Score', 'Execution_Decision']])
                    st.html(ai_output)
    except KeyError as e:
        st.error(f"Please ensure the column names in your code match the actual headers in your CSV files. Missing/Mismatch Error: {e}")
elif(selected == 'Price Discovery'):
    st.header('Price Discovery')
    st.write("Instructional Text for Price Discovery")

    # YOUR CODE FOR PRICE DISCOVERY GOES HERE
elif(selected == 'Regulatory and Compliance'):
    st.header('Regulatory and Compliance')
    st.write("Instructional Text for Regulatory and Compliance")

    def chunk_text(text: str, chunk_size=800, chunk_overlap=100) -> list[str]:
        """Splits text into overlapping chunks using smart semantic separators."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        return text_splitter.split_text(text)

    def get_relevant_context(query: str, chunks: list[str], top_k: int = 3) -> str:
        """Embeds chunks and query to retrieve top matches via cosine similarity."""
        
        # Initialize a fast, local embedding model (e.g., 384 dimensions)
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Convert query and text chunks into vector embeddings
        query_vector = embedding_model.encode([query])
        chunk_vectors = embedding_model.encode(chunks)

        # Compute cosine similarities manually (alternative to installing full FAISS db)
        import numpy as np

        similarities = np.dot(chunk_vectors, query_vector.T).squeeze()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Combine the best matching text chunks into a unified context string
        retrieved_chunks = [chunks[i] for i in top_indices]
        return "\n---\n".join(retrieved_chunks)

    def query_deepseek(query: str, context: str) -> str:
        """Sends user query accompanied by retrieved PDF context to DeepSeek."""
        system_prompt = (
            "You are an AI assistant. Answer the user prompt strictly using the "
            "provided PDF document context. If you don't know, say you don't know."
        )

        user_prompt = f"Context from PDF:\n{context}\n\nQuestion: {query}"

        llm = OpenAI(
            api_key=st.secrets["OPENAI_API_KEY"], base_url="https://api.deepseek.com"
        )

        response = llm.chat.completions.create(
            model="deepseek-chat",  # For general QA / deepseek-reasoning for R1
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,  # Low temperature forces truthfulness to the context
        )
        return response.choices[0].message.content

    def query_deepseek_structured(query: str, context: str) -> AnotherDocSum:
        """Uses DeepSeek's supported json_object mode, injecting schema targets in the prompt."""
        
        # Generate the target JSON schema block dynamically from the Pydantic model
        schema_json = json.dumps(AnotherDocSum.model_json_schema(), indent=2)
        
        # Rule 1: Always include the word 'json' explicitly in your prompt when using json_object mode
        system_prompt = (
            "You are an expert extraction assistant. You must respond ONLY with a valid JSON object. "
            f"The JSON output must strictly comply with this structural schema:\n{schema_json}"
        )

        user_prompt = f"Context from PDF:\n{context}\n\nQuestion: {query}"

        # Rule 2 - Use standard chat.completions.create with the correct json_object dictionary format
        deepseek_client = OpenAI(
            api_key=st.secrets["OPENAI_API_KEY"], base_url="https://api.deepseek.com"
        )

        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},  # Validated format allowed by DeepSeek
            temperature=0.1,
        )
        
        raw_content = response.choices[0].message.content
        
        return raw_content

        # Rule 3: Safely parse and validate the JSON data locally against Pydantic
        #try:
        #    parsed_json = json.loads(raw_content)
            
        #    # Validates fields and returns a fully verified Pydantic model instance
        #    return DocumentSummary.model_validate(parsed_json)
        #except (json.JSONDecodeError, ValidationError) as e:
        #    return raw_content

    selected = st.radio(
        "Choose an option:",
        ["Regulatory and Compliance - Simple Query", "Regulatory and Compliance - Dashboard"]
    )

    if(selected == "Regulatory and Compliance - Simple Query"):
        # File uploader restricted to PDFs
        uploaded_files = st.file_uploader(
            "Upload 3 PDF files", 
            type=["pdf"], 
            accept_multiple_files=True
        )

        # 3. Process files when exactly 3 are uploaded
        if uploaded_files:
            if st.button("Process & Generate JSON", type="primary"):
                with st.spinner("Extracting text from PDFs and invoking LLM..."):
                    try:
                        combined_text = ""
                        
                        # 1 - Extract text from each PDF
                        for uploaded_file in uploaded_files:
                            pdf_reader = PdfReader(uploaded_file)
                            combined_text += f"\n--- Start of Document: {uploaded_file.name} ---\n"
                            for page in pdf_reader.pages:
                                text = page.extract_text()
                                if text:
                                    combined_text += text
                            combined_text += f"\n--- End of Document: {uploaded_file.name} ---\n"

                        # 2 - Chunk the Text
                        document_chunks = chunk_text(combined_text)

                        # Execution Query
                        user_query_2 = 'What are the registration requirements for injectable antibiotics in Kenya?'
                        user_query_3 = 'What local representation is required in Kenya?'
                        user_query_4 = 'Compare Kenya regulatory complexity vs Vietnam'
                        user_query_5 = 'What approvals are required before importation?'
                        
                        user_query = user_query_5

                        # 3 - Calculating embeddings and matching context
                        matched_context = get_relevant_context(user_query, document_chunks, top_k=3)

                        # 4 - Requesting Structured JSON from DeepSeek
                        ai_answer = query_deepseek(user_query, matched_context)                    

                        st.write(ai_answer)
                    except Exception as e:
                        st.error(f"An error occurred during processing: {str(e)}")
    elif (selected == "Regulatory and Compliance - Dashboard"):
        # 1. Define the desired JSON structure using Pydantic
        class DocumentSummaryList(BaseModel):
            regions: list[str] = Field(description="List of titles or names of the processed PDFs.")
            registration_complexity: list[str] = Field(description="Registration complexity")
            safety_mandates: list[str] = Field(description="Safety mandates")
            parallel_import_rules: list[str] = Field(description="Import rules")
            risk_score: list[str] = Field(description="Risk score per region")
        
        class DocumentSummary(BaseModel):
            country: str = Field(description="Name of country on which the document is based")
            registration_complexity: str = Field(description="Registration Complexity for that country")
            safety_ai_mandates: str = Field(description="Safety AI Mandates")
            parallel_import_rules: str = Field(description=" Parallel Import Rules")
            risk_score: str = Field(description="Risk Score for the concerned country")
        
        class AnotherDocSum(BaseModel):
            country: str = Field(description="Name of country on which the document is based")
            foreign_data_acceptability: str = Field(description="Foreign Data Acceptability")
            price_determination: str = Field(description="Price Determination")

        # File uploader restricted to PDFs
        uploaded_files = st.file_uploader(
            "Upload 3 PDF files", 
            type=["pdf"], 
            accept_multiple_files=True
        )

        # 3. Process files when exactly 3 are uploaded
        if uploaded_files:
            #if len(uploaded_files) != 3:
            #    st.warning(f"Please upload exactly 3 files. Currently uploaded: {len(uploaded_files)}")
            #else:
            if st.button("Process & Generate JSON", type="primary"):
                with st.spinner("Extracting text from PDFs and invoking LLM..."):
                    try:
                        combined_text = ""
                        
                        # 1 - Extract text from each PDF
                        for uploaded_file in uploaded_files:
                            pdf_reader = PdfReader(uploaded_file)
                            combined_text += f"\n--- Start of Document: {uploaded_file.name} ---\n"
                            for page in pdf_reader.pages:
                                text = page.extract_text()
                                if text:
                                    combined_text += text
                            combined_text += f"\n--- End of Document: {uploaded_file.name} ---\n"

                        # 2 - Chunk the Text
                        document_chunks = chunk_text(combined_text)

                        # Execution Query
                        user_query = "Analyze the text read from the PDFs and derive output as per the required JSON structure"
                        
                        # 3 - Calculating embeddings and matching context
                        matched_context = get_relevant_context(user_query, document_chunks, top_k=3)

                        # 4 - Requesting Structured JSON from DeepSeek
                        structured_data = query_deepseek_structured(user_query, matched_context)                    

                        # Provide a direct download button for the JSON file
                        json_string = json.dumps(structured_data, indent=4)
                        st.download_button(
                            label="Download JSON File",
                            file_name="pdf_analysis.json",
                            mime="application/json",
                            data=json_string
                        )
                        #st.write(structured_data.model_dump_json(indent=2))

                    except Exception as e:
                        st.error(f"An error occurred during processing: {str(e)}")
