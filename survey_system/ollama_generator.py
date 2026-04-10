"""
Ollama-based survey response generator for AI residents.
Uses local Ollama LLM to generate contextual survey responses based on:
- Agent background information from agent.json
- Activity history from simulation.md
- Prompt templates from data/prompts/
"""

import os
import json
import requests
from typing import Dict, List, Optional, Any
from pathlib import Path
from string import Template
from dotenv import load_dotenv

load_dotenv()


class OllamaSurveyGenerator:
    """Generate survey responses using Ollama LLM with prompt templates."""

    def __init__(self, simulation_md_path: Optional[str] = None, agents_dir: Optional[str] = None):
        """
        Initialize Ollama generator.

        Args:
            simulation_md_path: Path to simulation.md file with activity history
            agents_dir: Path to agents directory (default: frontend/static/assets/village/agents)
        """
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        self.api_url = f"{self.base_url}/api/generate"

        # Set paths
        self.project_root = Path(__file__).parent.parent
        self.prompts_dir = self.project_root / "data" / "prompts"

        # Set agents directory
        if agents_dir:
            self.agents_dir = Path(agents_dir)
        else:
            self.agents_dir = self.project_root / "frontend" / "static" / "assets" / "village" / "agents"

        # Load simulation history if provided
        self.simulation_history = {}
        if simulation_md_path and os.path.exists(simulation_md_path):
            self.simulation_history = self._load_simulation_history(simulation_md_path)
            print(f"✓ 已載入 {len(self.simulation_history)} 位居民的活動歷史")

        # Cache for agent data
        self.agents_cache = {}

    def _load_agent_data(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """
        Load agent data from agent.json file.

        Args:
            agent_name: Name of the agent

        Returns:
            Dictionary with agent data or None if not found
        """
        # Check cache first
        if agent_name in self.agents_cache:
            return self.agents_cache[agent_name]

        # Find agent directory
        agent_dir = self.agents_dir / agent_name
        agent_json_path = agent_dir / "agent.json"

        if not agent_json_path.exists():
            print(f"❌ 找不到 agent.json: {agent_json_path}")
            return None

        try:
            with open(agent_json_path, 'r', encoding='utf-8') as f:
                agent_data = json.load(f)

            # Cache the data
            self.agents_cache[agent_name] = agent_data
            return agent_data

        except Exception as e:
            print(f"❌ 載入 {agent_name} 的資料時發生錯誤: {e}")
            return None

    def _load_simulation_history(self, file_path: str) -> Dict[str, str]:
        """
        Load and parse simulation.md file to extract activity history for each resident.

        Args:
            file_path: Path to simulation.md file

        Returns:
            Dictionary mapping resident names to their activity summaries
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            history_dict = {}
            lines = content.split('\n')

            current_resident = None
            resident_activities = []

            for line in lines:
                # Check if this is a resident header (### 居民名)
                if line.startswith('### '):
                    # Save previous resident's activities
                    if current_resident and resident_activities:
                        # Keep last 100 activities for context
                        history_dict[current_resident] = '\n'.join(resident_activities[-100:])

                    # Start new resident
                    current_resident = line.replace('### ', '').strip()
                    resident_activities = []

                # Collect activity and location info
                elif current_resident and (line.startswith('位置：') or line.startswith('活動：')):
                    resident_activities.append(line)

            # Save last resident
            if current_resident and resident_activities:
                history_dict[current_resident] = '\n'.join(resident_activities[-100:])

            return history_dict

        except Exception as e:
            print(f"⚠️  載入活動歷史失敗: {e}")
            return {}

    def _load_prompt_template(self, question_type: str) -> Optional[Template]:
        """
        Load prompt template from file.

        Args:
            question_type: Type of question (single_choice, multiple_choice, rating, text)

        Returns:
            Template object or None if file not found
        """
        template_file = self.prompts_dir / f"survey_{question_type}.txt"

        if not template_file.exists():
            print(f"⚠️  找不到 prompt 模板: {template_file}")
            return None

        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                template_content = f.read()
            return Template(template_content)

        except Exception as e:
            print(f"❌ 載入 prompt 模板時發生錯誤: {e}")
            return None

    def _format_options(self, options: List[str]) -> str:
        """
        Format options for prompt.

        Args:
            options: List of option strings

        Returns:
            Formatted options string
        """
        if not options:
            return ""

        formatted = []
        for i, option in enumerate(options, 1):
            formatted.append(f"{i}. {option}")

        return "\n".join(formatted)

    def _build_prompt(
        self,
        agent_name: str,
        agent_data: Dict[str, Any],
        question_text: str,
        question_type: str,
        options: Optional[List[str]] = None
    ) -> str:
        """
        Build prompt using template and agent data.

        Args:
            agent_name: Name of the agent
            agent_data: Agent data from agent.json
            question_text: The survey question text
            question_type: Type of question
            options: List of answer options

        Returns:
            Formatted prompt string
        """
        # Load template
        template = self._load_prompt_template(question_type)
        if not template:
            return f"無法載入 {question_type} 的 prompt 模板"

        # Extract data from agent.json
        scratch = agent_data.get("scratch", {})

        # Prepare template variables
        template_vars = {
            "agent_name": agent_name,
            "age": scratch.get("age", "未知"),
            "personality": scratch.get("innate", "未知"),
            "interests": scratch.get("learned", "未知"),
            "lifestyle": scratch.get("lifestyle", "未知"),
            "current_activity": agent_data.get("currently", "未知"),
            "family_background": scratch.get("family_background", "無相關資訊"),
            "wealth_level": scratch.get("wealth_level", "無相關資訊"),
            "question_text": question_text,
        }

        # Add activity history if available
        if agent_name in self.simulation_history:
            activity_history = self.simulation_history[agent_name]
            # Limit to last 1000 characters
            template_vars["activity_history"] = activity_history[-1000:] if len(activity_history) > 1000 else activity_history
        else:
            template_vars["activity_history"] = "無活動記錄"

        # Add options for choice questions
        if options and question_type in ['single_choice', 'multiple_choice']:
            template_vars["options"] = self._format_options(options)
        else:
            template_vars["options"] = ""

        # Substitute template variables
        try:
            prompt = template.safe_substitute(template_vars)
            return prompt
        except Exception as e:
            print(f"❌ 替換 prompt 變數時發生錯誤: {e}")
            return f"模板處理錯誤: {e}"

    def generate_response(
        self,
        resident_name: str,
        resident_info: Dict[str, Any],  # 保留此參數以兼容現有接口，但不使用
        question_text: str,
        question_type: str,
        options: Optional[List[str]] = None
    ) -> str:
        """
        Generate a survey response using Ollama LLM.

        Args:
            resident_name: Name of the resident
            resident_info: Dictionary containing resident background info (NOT USED - for compatibility)
            question_text: The survey question text
            question_type: Type of question (single_choice, multiple_choice, rating, text)
            options: List of answer options (for choice questions)

        Returns:
            Generated response text
        """
        # Load agent data from JSON file
        agent_data = self._load_agent_data(resident_name)
        if not agent_data:
            return self._fallback_response(question_type)

        # Build prompt using template
        prompt = self._build_prompt(
            agent_name=resident_name,
            agent_data=agent_data,
            question_text=question_text,
            question_type=question_type,
            options=options
        )

        # Call Ollama API
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 500
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                print(f"❌ Ollama API 錯誤: {response.status_code} - {response.text}")
                return self._fallback_response(question_type)

        except Exception as e:
            print(f"❌ 調用 Ollama API 時發生錯誤: {e}")
            return self._fallback_response(question_type)

    def _fallback_response(self, question_type: str) -> str:
        """
        Provide fallback response when Ollama API fails.

        Args:
            question_type: Type of question

        Returns:
            Default fallback response
        """
        fallbacks = {
            'single_choice': '其他',
            'multiple_choice': '其他',
            'rating': '5',
            'text': '抱歉，目前無法提供詳細回答。'
        }
        return fallbacks.get(question_type, '無回答')

    def batch_generate_responses(
        self,
        resident_name: str,
        resident_info: Dict[str, Any],
        questions: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate responses for multiple questions in batch.

        Args:
            resident_name: Name of the resident
            resident_info: Dictionary containing resident background info
            questions: List of question dictionaries

        Returns:
            List of generated responses
        """
        responses = []

        for question in questions:
            response = self.generate_response(
                resident_name=resident_name,
                resident_info=resident_info,
                question_text=question.get('question_text', ''),
                question_type=question.get('question_type', 'text'),
                options=question.get('options', [])
            )
            responses.append(response)

        return responses
