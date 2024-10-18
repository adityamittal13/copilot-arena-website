import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import aggregation
from privacy import PrivacySetting, clean_data
import pandas as pd


class FirebaseClient:
    def __init__(self):
        firebase_config_path = "firebase-config.json"
        self.cred = credentials.Certificate(firebase_config_path)
        firebase_admin.initialize_app(self.cred)
        self.db = firestore.client()

    def upload_data(self, collection_name: str, data: dict, privacy: PrivacySetting):
        collection_ref = self.db.collection(collection_name)
        if privacy != PrivacySetting.RESEARCH:
            data = clean_data(data)
        collection_ref.add(data)

    def get_autocomplete_outcomes(self, collection_name: str, user_id: str = None):
        query = self.db.collection(collection_name)
        if user_id:
            query = query.where(filter=firestore.FieldFilter("userId", "==", user_id))
        outcomes = query.get()
        outcomes_df = pd.DataFrame([x.to_dict() for x in outcomes])
        return outcomes_df

    def get_autocomplete_completions(
        self, collection_name: str, models: list, user_id: str = None
    ):
        query = self.db.collection(collection_name)
        filters = [firestore.FieldFilter("model", "in", models)]

        if user_id:
            filters.append(firestore.FieldFilter("userId", "==", user_id))

        query = query.where(filter=firestore.And(filters))

        completions = query.get()
        completions_df = pd.DataFrame([x.to_dict() for x in completions])
        return completions_df

    def get_autocomplete_outcomes_count(
        self, collection_name: str, user_id: str = None
    ):
        query = self.db.collection(collection_name)
        if user_id:
            query = query.where(filter=firestore.FieldFilter("userId", "==", user_id))

        aggregate_query = aggregation.AggregationQuery(query)
        aggregate_query.count(alias="all")

        results = aggregate_query.get()
        return results[0][0].value