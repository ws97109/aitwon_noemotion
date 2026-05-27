"""
問卷系統核心數據模型
定義問卷、回應和管理器類別
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional


class Survey:
    """問卷類別"""
    
    def __init__(self, survey_id: str = None, title: str = "", description: str = ""):
        self.survey_id = survey_id or str(uuid.uuid4())
        self.title = title
        self.description = description
        self.questions = []
        self.created_at = datetime.now().isoformat()
        
    def add_question(self, question_type: str, question_text: str, 
                    options: Optional[List[str]] = None, required: bool = True):
        """新增問題
        
        Args:
            question_type: 問題類型 ('single_choice', 'multiple_choice', 'text', 'rating')
            question_text: 問題內容
            options: 選項列表（單選/多選題使用）
            required: 是否必填
        """
        question = {
            "id": len(self.questions) + 1,
            "type": question_type,
            "text": question_text,
            "options": options or [],
            "required": required
        }
        self.questions.append(question)
        return question["id"]
    
    def to_dict(self) -> Dict:
        """轉換為字典格式"""
        return {
            "survey_id": self.survey_id,
            "title": self.title,
            "description": self.description,
            "questions": self.questions,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Survey':
        """從字典創建問卷實例"""
        survey = cls(
            survey_id=data["survey_id"],
            title=data["title"], 
            description=data.get("description", "")
        )
        survey.questions = data["questions"]
        survey.created_at = data["created_at"]
        return survey


class SurveyResponse:
    """問卷回應類別"""
    
    def __init__(self, survey_id: str, respondent_name: str):
        self.response_id = str(uuid.uuid4())
        self.survey_id = survey_id
        self.respondent_name = respondent_name
        self.responses = {}
        self.completed_at = None
        self.started_at = datetime.now().isoformat()
    
    def add_response(self, question_id: int, answer: Any):
        """新增答案"""
        self.responses[str(question_id)] = answer
    
    def complete(self):
        """標記為完成"""
        self.completed_at = datetime.now().isoformat()
    
    def is_completed(self) -> bool:
        """檢查是否完成"""
        return self.completed_at is not None
    
    def to_dict(self) -> Dict:
        """轉換為字典格式"""
        return {
            "response_id": self.response_id,
            "survey_id": self.survey_id,
            "respondent_name": self.respondent_name,
            "responses": self.responses,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SurveyResponse':
        """從字典創建回應實例"""
        response = cls(data["survey_id"], data["respondent_name"])
        response.response_id = data["response_id"]
        response.responses = data["responses"]
        response.started_at = data["started_at"]
        response.completed_at = data.get("completed_at")
        return response


class SurveyManager:
    """問卷管理器"""
    
    def __init__(self, storage_path: str = "survey_system/data"):
        self.storage_path = storage_path
        self._init_storage()
    
    def _init_storage(self):
        """初始化存儲目錄"""
        os.makedirs(self.storage_path, exist_ok=True)
        os.makedirs(f"{self.storage_path}/surveys", exist_ok=True)
        os.makedirs(f"{self.storage_path}/responses", exist_ok=True)
        os.makedirs(f"{self.storage_path}/exports", exist_ok=True)
    
    def save_survey(self, survey: Survey) -> bool:
        """儲存問卷"""
        try:
            survey_file = f"{self.storage_path}/surveys/{survey.survey_id}.json"
            with open(survey_file, 'w', encoding='utf-8') as f:
                json.dump(survey.to_dict(), f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存問卷失敗: {e}")
            return False
    
    def load_survey(self, survey_id: str) -> Optional[Survey]:
        """載入問卷"""
        try:
            survey_file = f"{self.storage_path}/surveys/{survey_id}.json"
            if not os.path.exists(survey_file):
                return None
            
            with open(survey_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return Survey.from_dict(data)
        except Exception as e:
            print(f"載入問卷失敗: {e}")
            return None
    
    def get_all_surveys(self) -> List[Dict]:
        """取得所有問卷列表"""
        surveys = []
        surveys_dir = f"{self.storage_path}/surveys"
        
        if not os.path.exists(surveys_dir):
            return surveys
            
        try:
            for filename in os.listdir(surveys_dir):
                if filename.endswith('.json'):
                    survey_id = filename[:-5]
                    survey = self.load_survey(survey_id)
                    if survey:
                        surveys.append({
                            "survey_id": survey.survey_id,
                            "title": survey.title,
                            "description": survey.description,
                            "created_at": survey.created_at,
                            "question_count": len(survey.questions)
                        })
        except Exception as e:
            print(f"獲取問卷列表失敗: {e}")
        
        return surveys
    
    def delete_survey(self, survey_id: str) -> bool:
        """刪除問卷"""
        try:
            survey_file = f"{self.storage_path}/surveys/{survey_id}.json"
            if os.path.exists(survey_file):
                os.remove(survey_file)
                
                # 同時刪除相關回應
                self.delete_survey_responses(survey_id)
                return True
            return False
        except Exception as e:
            print(f"刪除問卷失敗: {e}")
            return False
    
    def save_response(self, response: SurveyResponse) -> bool:
        """儲存問卷回應"""
        try:
            response_file = f"{self.storage_path}/responses/{response.response_id}.json"
            with open(response_file, 'w', encoding='utf-8') as f:
                json.dump(response.to_dict(), f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存回應失敗: {e}")
            return False
    
    def get_responses_by_survey(self, survey_id: str) -> List[SurveyResponse]:
        """根據問卷ID取得所有回應"""
        responses = []
        responses_dir = f"{self.storage_path}/responses"
        
        if not os.path.exists(responses_dir):
            return responses
            
        try:
            for filename in os.listdir(responses_dir):
                if filename.endswith('.json'):
                    response_file = f"{responses_dir}/{filename}"
                    with open(response_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if data["survey_id"] == survey_id:
                        responses.append(SurveyResponse.from_dict(data))
        except Exception as e:
            print(f"獲取回應失敗: {e}")
        
        return responses
    
    def delete_survey_responses(self, survey_id: str) -> int:
        """刪除問卷的所有回應"""
        deleted_count = 0
        responses_dir = f"{self.storage_path}/responses"

        if not os.path.exists(responses_dir):
            return deleted_count

        try:
            for filename in os.listdir(responses_dir):
                if filename.endswith('.json'):
                    response_file = f"{responses_dir}/{filename}"
                    with open(response_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    if data["survey_id"] == survey_id:
                        os.remove(response_file)
                        deleted_count += 1
        except Exception as e:
            print(f"刪除回應失敗: {e}")

        return deleted_count

    def delete_resident_responses(self, survey_id: str, respondent_name: str) -> int:
        """刪除特定問卷中特定居民的所有回應（用於避免重複累積）"""
        deleted_count = 0
        responses_dir = f"{self.storage_path}/responses"

        if not os.path.exists(responses_dir):
            return deleted_count

        try:
            for filename in os.listdir(responses_dir):
                if not filename.endswith('.json'):
                    continue
                response_file = f"{responses_dir}/{filename}"
                try:
                    with open(response_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception:
                    continue
                if data.get("survey_id") == survey_id and data.get("respondent_name") == respondent_name:
                    os.remove(response_file)
                    deleted_count += 1
        except Exception as e:
            print(f"刪除居民回應失敗: {e}")

        return deleted_count
    
    def get_survey_stats(self, survey_id: str) -> Dict:
        """取得問卷統計信息"""
        survey = self.load_survey(survey_id)
        if not survey:
            return {}
        
        responses = self.get_responses_by_survey(survey_id)
        completed_responses = [r for r in responses if r.is_completed()]
        
        return {
            "survey_id": survey_id,
            "title": survey.title,
            "question_count": len(survey.questions),
            "total_responses": len(responses),
            "completed_responses": len(completed_responses),
            "completion_rate": len(completed_responses) / len(responses) * 100 if responses else 0,
            "created_at": survey.created_at
        }
    
    def duplicate_survey(self, survey_id: str, new_title: str = None) -> Optional[str]:
        """複製問卷"""
        try:
            original_survey = self.load_survey(survey_id)
            if not original_survey:
                return None
            
            new_survey = Survey(
                title=new_title or f"{original_survey.title} (複製)",
                description=original_survey.description
            )
            
            # 複製所有問題
            for question in original_survey.questions:
                new_survey.add_question(
                    question_type=question["type"],
                    question_text=question["text"],
                    options=question.get("options"),
                    required=question.get("required", True)
                )
            
            if self.save_survey(new_survey):
                return new_survey.survey_id
            return None
            
        except Exception as e:
            print(f"複製問卷失敗: {e}")
            return None
    
    def batch_delete_surveys(self, survey_ids: List[str]) -> Dict[str, bool]:
        """批量刪除問卷"""
        results = {}
        for survey_id in survey_ids:
            results[survey_id] = self.delete_survey(survey_id)
        return results
    
    def update_survey(self, survey_id: str, title: str = None, description: str = None) -> bool:
        """更新問卷基本信息"""
        try:
            survey = self.load_survey(survey_id)
            if not survey:
                return False
            
            if title is not None:
                survey.title = title
            if description is not None:
                survey.description = description
            
            return self.save_survey(survey)
            
        except Exception as e:
            print(f"更新問卷失敗: {e}")
            return False
    
    def archive_survey(self, survey_id: str) -> bool:
        """歸檔問卷（移動到歸檔目錄）"""
        try:
            archive_dir = f"{self.storage_path}/archived"
            os.makedirs(archive_dir, exist_ok=True)
            
            survey_file = f"{self.storage_path}/surveys/{survey_id}.json"
            archive_file = f"{archive_dir}/{survey_id}.json"
            
            if os.path.exists(survey_file):
                os.rename(survey_file, archive_file)
                return True
            return False
            
        except Exception as e:
            print(f"歸檔問卷失敗: {e}")
            return False
    
    def search_surveys(self, keyword: str) -> List[Dict]:
        """搜索問卷"""
        all_surveys = self.get_all_surveys()
        keyword_lower = keyword.lower()
        
        matching_surveys = []
        for survey in all_surveys:
            if (keyword_lower in survey["title"].lower() or 
                keyword_lower in survey["description"].lower()):
                matching_surveys.append(survey)
        
        return matching_surveys
    
    def get_response_analytics(self, survey_id: str) -> Dict:
        """獲取問卷回應詳細分析"""
        survey = self.load_survey(survey_id)
        if not survey:
            return {}
        
        responses = self.get_responses_by_survey(survey_id)
        completed_responses = [r for r in responses if r.is_completed()]
        
        analytics = {
            "survey_info": {
                "survey_id": survey_id,
                "title": survey.title,
                "question_count": len(survey.questions)
            },
            "response_summary": {
                "total_responses": len(responses),
                "completed_responses": len(completed_responses),
                "completion_rate": len(completed_responses) / len(responses) * 100 if responses else 0
            },
            "question_analytics": {}
        }
        
        # 分析每個問題的回應
        for question in survey.questions:
            question_id = str(question["id"])
            question_type = question["type"]

            question_responses = []
            named_responses = []  # 文字題用：[(respondent_name, answer), ...]
            for response in completed_responses:
                if question_id in response.responses:
                    question_responses.append(response.responses[question_id])
                    named_responses.append((response.respondent_name, response.responses[question_id]))

            if question_type == "single_choice":
                analytics["question_analytics"][question_id] = self._analyze_choice_question(
                    question_responses, question.get("options", [])
                )
            elif question_type == "multiple_choice":
                analytics["question_analytics"][question_id] = self._analyze_multiple_choice_question(
                    question_responses, question.get("options", [])
                )
            elif question_type == "rating":
                analytics["question_analytics"][question_id] = self._analyze_rating_question(
                    question_responses
                )
            elif question_type == "text":
                analytics["question_analytics"][question_id] = self._analyze_text_question(
                    question_responses, named_responses
                )
        
        return analytics
    
    def _analyze_choice_question(self, responses: List[str], options: List[str]) -> Dict:
        """分析單選題回應"""
        choice_counts = {}
        for option in options:
            choice_counts[option] = 0
        
        for response in responses:
            if response in choice_counts:
                choice_counts[response] += 1
        
        total = len(responses)
        percentages = {}
        for option, count in choice_counts.items():
            percentages[option] = (count / total * 100) if total > 0 else 0
        
        return {
            "type": "single_choice",
            "total_responses": total,
            "choice_counts": choice_counts,
            "percentages": percentages
        }
    
    def _analyze_multiple_choice_question(self, responses: List[List[str]], options: List[str]) -> Dict:
        """分析多選題回應"""
        choice_counts = {}
        for option in options:
            choice_counts[option] = 0
        
        total_responses = len(responses)
        for response_list in responses:
            if isinstance(response_list, list):
                for choice in response_list:
                    if choice in choice_counts:
                        choice_counts[choice] += 1
        
        percentages = {}
        for option, count in choice_counts.items():
            percentages[option] = (count / total_responses * 100) if total_responses > 0 else 0
        
        return {
            "type": "multiple_choice",
            "total_responses": total_responses,
            "choice_counts": choice_counts,
            "percentages": percentages
        }
    
    def _analyze_rating_question(self, responses: List[int]) -> Dict:
        """分析評分題回應"""
        if not responses:
            return {
                "type": "rating",
                "total_responses": 0,
                "average": 0,
                "min": 0,
                "max": 0,
                "distribution": {}
            }
        
        numeric_responses = []
        for response in responses:
            try:
                numeric_responses.append(int(response))
            except (ValueError, TypeError):
                continue
        
        if not numeric_responses:
            return {
                "type": "rating",
                "total_responses": 0,
                "average": 0,
                "min": 0,
                "max": 0,
                "distribution": {}
            }
        
        distribution = {}
        for i in range(1, 11):  # 1-10分
            distribution[str(i)] = numeric_responses.count(i)
        
        return {
            "type": "rating",
            "total_responses": len(numeric_responses),
            "average": sum(numeric_responses) / len(numeric_responses),
            "min": min(numeric_responses),
            "max": max(numeric_responses),
            "distribution": distribution
        }
    
    def _analyze_text_question(self, responses: List[str], named_responses=None) -> Dict:
        """分析文字題回應

        Args:
            responses: 答案字串列表
            named_responses: [(respondent_name, answer), ...] — 供前端按人列出
        """
        if not responses:
            return {
                "type": "text",
                "total_responses": 0,
                "average_length": 0,
                "common_words": [],
                "answers_by_person": []
            }

        total_length = sum(len(str(response)) for response in responses)
        average_length = total_length / len(responses)

        # 簡單的詞頻分析
        word_counts = {}
        for response in responses:
            words = str(response).split()
            for word in words:
                word = word.lower().strip('.,!?')
                if len(word) > 2:  # 只統計長度大於2的詞
                    word_counts[word] = word_counts.get(word, 0) + 1

        # 取前10個最常見的詞
        common_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # 每人對應的回答（按姓名排序維持穩定）
        answers_by_person = []
        if named_responses:
            answers_by_person = [
                {"name": name, "answer": str(answer)}
                for name, answer in named_responses
            ]

        return {
            "type": "text",
            "total_responses": len(responses),
            "average_length": average_length,
            "common_words": common_words,
            "answers_by_person": answers_by_person
        }