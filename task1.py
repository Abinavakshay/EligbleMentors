import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

# Mock Data for Mentors
mentors = pd.DataFrame({
    'mentor_id': [1, 2, 3, 4, 5],
    'name': ['Aditi', 'Rahul', 'Sneha', 'Karan', 'Fatima'],
    'preferred_subjects': [['Legal Aptitude', 'GK'], ['English', 'GK'], ['Legal Aptitude'], ['GK'], ['English']],
    'target_colleges': [['NLSIU'], ['NALSAR'], ['NLU Delhi'], ['NLU Jodhpur'], ['NALSAR']],
    'prep_level': ['Advanced', 'Advanced', 'Intermediate', 'Beginner', 'Advanced'],
    'learning_style': ['Visual', 'Reading/Writing', 'Auditory', 'Kinesthetic', 'Visual']
})

# Input Aspirant Profile
aspirant_profile = {
    'preferred_subjects': ['Legal Aptitude', 'GK'],
    'target_colleges': ['NLSIU'],
    'prep_level': 'Beginner',
    'learning_style': 'Visual'
}

# Combine Data
full_data = mentors.copy()
full_data.loc[len(full_data.index)] = ['aspirant', 'Aspirant'] + list(aspirant_profile.values())

# Feature Encoding
mlb = MultiLabelBinarizer()

# Encode preferred_subjects and target_colleges
subjects_encoded = mlb.fit_transform(full_data['preferred_subjects'])
subjects_df = pd.DataFrame(subjects_encoded, columns=mlb.classes_)

colleges_encoded = mlb.fit_transform(full_data['target_colleges'])
colleges_df = pd.DataFrame(colleges_encoded, columns=mlb.classes_)

# Encode prep_level and learning_style using one-hot
prep_encoded = pd.get_dummies(full_data['prep_level'], prefix='level')
style_encoded = pd.get_dummies(full_data['learning_style'], prefix='style')

# Concatenate all features
features = pd.concat([subjects_df, colleges_df, prep_encoded, style_encoded], axis=1)

# Calculate Cosine Similarity
similarities = cosine_similarity(features)
aspirant_similarities = similarities[-1][:-1]  # Exclude aspirant vs aspirant

# Recommend Top 3 Mentors
top_indices = aspirant_similarities.argsort()[-3:][::-1]
recommendations = mentors.iloc[top_indices]

print("Top 3 Recommended Mentors:")
print(recommendations[['name', 'preferred_subjects', 'target_colleges']])
