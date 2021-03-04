import os
import pandas as pd

directory_path = "/mnt/nfs/scratch1/neerajsharma/amazon_data"

data_files = [("reviews_Beauty_5.json.gz", "ratings_Beauty.csv"),
              ("reviews_Cell_Phones_and_Accessories_5.json.gz", "ratings_Cell_Phones_and_Accessories.csv"),
              ("reviews_Health_and_Personal_Care_5.json.gz", "ratings_Health_and_Personal_Care.csv")]

for review_data, rating_data in data_files:
    review_file_path = os.path.join(directory_path, review_data)
    rating_file_path = os.path.join(directory_path, rating_data)
    reviews_df = pd.read_json(review_file_path, lines=True)
    reviews_df.drop(['reviewerName', 'unixReviewTime', 'reviewTime'], axis=1, inplace=True)
    ratings_df = pd.read_csv(rating_file_path, header=None)
