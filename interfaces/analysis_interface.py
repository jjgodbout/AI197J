import streamlit as st
import pandas as pd
from typing import Dict, Optional, List
from context.analyzer import TextAnalyzer
from logging import getLogger

logger = getLogger(__name__)


class AnalysisInterface:
    AVAILABLE_MODELS = [
        "gemma-7b",
        "mixtral-8x7b"
    ]

    MODEL_DESCRIPTIONS = {
        "gemma-7b": "Google's lightweight yet capable model",
        "mixtral-8x7b": "Mixture of experts model"
    }

    def __init__(self, get_user_documents_func):
        """
        Initialize Analysis Interface

        Args:
            get_user_documents_func: Function to get user documents from the manager
        """
        self.get_user_documents = get_user_documents_func

    def _create_model_selector(self) -> str:
        """Create an enhanced model selector with descriptions"""
        col1, col2 = st.columns([1, 2])

        with col1:
            selected_model = st.selectbox(
                "Select model",
                options=self.AVAILABLE_MODELS,
                key="model_selector",
                help="Choose the AI model for completion analysis"
            )

        with col2:
            if selected_model:
                st.info(self.MODEL_DESCRIPTIONS.get(selected_model, ""))

        return selected_model

    def render(self):
        """Render the text analysis interface"""
        st.subheader("Document Analysis")

        # Get user email from session
        user_email = st.session_state.get("username")
        if not user_email:
            st.error("Please log in to analyze documents")
            return

        # Get documents
        docs_df = self.get_user_documents(user_email)
        if docs_df.empty:
            st.info("No documents available for analysis.")
            return

        # Create document selection options
        doc_options = docs_df[['id', 'name']].copy()
        doc_dict = dict(zip(doc_options['name'], doc_options['id']))

        # Create columns for layout
        col1, col2 = st.columns([2, 1])

        with col1:
            # Document selection
            selected_doc = st.selectbox(
                "Select a document to analyze",
                options=list(doc_dict.keys()),
                key="analysis_doc_selector"
            )

            # Analysis method selection
            analysis_methods = {
                "Summary": "get_summary",
                "Classification": "classify_text",
                "Question Answering": "extract_answer",
                "Completion": "complete_analysis"
            }

            selected_method = st.selectbox(
                "Select analysis method",
                options=list(analysis_methods.keys()),
                key="analysis_method_selector"
            )

        with col2:
            # Page group size selection
            page_group_size = st.number_input(
                "Pages per group",
                min_value=1,
                max_value=50,
                value=5,
                help="Number of pages to analyze together"
            )

        # Initialize analyzer
        analyzer = TextAnalyzer(page_group_size=page_group_size)

        # Method-specific inputs
        st.divider()

        if selected_method == "Classification":
            categories = st.text_input(
                "Enter categories (comma-separated)",
                value="summary,detail",
                help="Categories for classification"
            ).split(',')
            categories = [cat.strip() for cat in categories]

        elif selected_method == "Question Answering":
            question = st.text_input(
                "Enter your question",
                help="Question to ask about the document"
            )

        elif selected_method == "Completion":
            prompt = st.text_area(
                "Enter your prompt",
                help="Prompt for completion analysis"
            )
            model = self._create_model_selector()

        # Analysis button
        if st.button("Run Analysis", type="primary"):
            if selected_doc:
                try:
                    document_id = str(doc_dict[selected_doc])
                    method = analysis_methods[selected_method]

                    with st.spinner(f"Running {selected_method.lower()} analysis..."):
                        results = self._run_analysis(
                            analyzer=analyzer,
                            method=selected_method,
                            document_id=document_id,
                            categories=categories if selected_method == "Classification" else None,
                            question=question if selected_method == "Question Answering" else None,
                            prompt=prompt if selected_method == "Completion" else None,
                            model=model if selected_method == "Completion" else None
                        )

                        if results:
                            st.success("Analysis completed!")

                            # Display results in expandable sections
                            if 'document' in results:
                                with st.expander("Document Information", expanded=True):
                                    st.json(results['document'])

                            if 'analysis' in results:
                                with st.expander("Analysis Results", expanded=True):
                                    st.json(results['analysis'])

                            if 'metadata' in results:
                                with st.expander("Analysis Metadata"):
                                    st.json(results['metadata'])
                        else:
                            st.warning("No results returned from analysis.")

                except Exception as e:
                    logger.error(f"Error during analysis: {str(e)}")
                    st.error(f"Error during analysis: {str(e)}")
            else:
                st.warning("Please select a document to analyze.")

    def _run_analysis(
            self,
            analyzer: TextAnalyzer,
            method: str,
            document_id: str,
            categories: Optional[List[str]] = None,
            question: Optional[str] = None,
            prompt: Optional[str] = None,
            model: Optional[str] = None
    ) -> Dict:
        """Run the selected analysis method with provided parameters"""

        if method == "Summary":
            return analyzer.get_summary(document_id)

        elif method == "Classification":
            if not categories:
                raise ValueError("Categories required for classification")
            return analyzer.classify_text(document_id, categories)

        elif method == "Question Answering":
            if not question:
                raise ValueError("Question required for question answering")
            return analyzer.extract_answer(document_id, question)

        elif method == "Completion":
            if not prompt:
                raise ValueError("Prompt required for completion")
            return analyzer.complete_analysis(
                doc_id=document_id,
                prompt=prompt,
                model=model or "mixtral-8x7b"
            )

        else:
            raise ValueError(f"Unknown analysis method: {method}")