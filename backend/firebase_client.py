import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd


class FirebaseClient:
    def __init__(self):
        firebase_config_path = "firebase-config.json"
        self.cred = credentials.Certificate(firebase_config_path)
        firebase_admin.initialize_app(self.cred)
        self.db = firestore.client()

    def get_autocomplete_outcomes(self, collection_name: str, user_id: str = None):
        query = self.db.collection(collection_name)
        if user_id:
            query = query.where(filter=firestore.FieldFilter("userId", "==", user_id))
        outcomes = query.get()
        outcomes_df = pd.DataFrame([x.to_dict() for x in outcomes])
        return outcomes_df