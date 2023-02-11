#%%
import pandas as pd


def read_csv_data(file):
    df = pd.read_csv(file)
    return df

def remove_rows_with_missing_ratings(df):
    df=df.dropna(subset=['Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating'])
    return df

def combine_description_strings(df):
    df = df.dropna(subset=['Description'])
    df["Description"] = df["Description"].apply(lambda x: x.replace("'About this space', ", '').replace("'', ", '').replace('[', '').replace(']', '').replace('\\n', '. ').replace("''", '').split(" "))
    df["Description"] = df["Description"].apply(lambda x: " ".join(x))
    return df

def set_default_feature_values(df):
    df.loc[:, ["guests", "bedrooms"]] = df.loc[:, ["guests", "bedrooms"]].fillna('1')
    df.loc[:, ["beds", "bathrooms"]] = df.loc[:, ["beds", "bathrooms"]].fillna(1)
    df.loc[df["guests"] == 'Somerford Keynes England United Kingdom', "guests"] = '1'
    df.loc[df["bedrooms"] == 'https://www.airbnb.co.uk/rooms/49009981?adults=1&category_tag=Tag%3A677&children=0&infants=0&search_mode=flex_destinations_search&check_in=2022-04-18&check_out=2022-04-25&previous_page_section_name=1000&federated_search_id=0b044c1c-8d17-4b03-bffb-5de13ff710bc', "bedrooms"] = '1'
    return df

def clean_tabular_data(df):
    df = remove_rows_with_missing_ratings(df)
    df = combine_description_strings(df)
    df = set_default_feature_values(df)
    return df

def load_airbnb(df):
    df = df.drop(columns=["Unnamed: 19"])
    df = df.drop(columns=["Unnamed: 0"])

    features = df.drop(columns=['Price_Night', 'ID', 'Category', 'Title', 'Description', 'Amenities', 'Location', 'url'])
    labels = df["Price_Night"]

    features_labels  = (features, labels)
    return features_labels


if __name__ == '__main__':

    # reads raw tabular data
    file = "tabular_data/listing.csv"
    df = read_csv_data(file)

    # cleans the tabular data
    df = clean_tabular_data(df)

    #writes the csv file
    df.to_csv('clean_tabular_data.csv')  

    df = read_csv_data('clean_tabular_data.csv')
    feature_labels = load_airbnb(df)
    feature, labels = feature_labels