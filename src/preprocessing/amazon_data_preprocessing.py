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
    reviews_summaries_per_user = reviews_df.groupby(by='reviewerID')['reviewText', 'summary']
    reviews_summaries_per_item = reviews_df.groupby(by='asin')['reviewText', 'summary']
    unique_users = reviews_summaries_per_user.groups.keys()
    unique_items = reviews_summaries_per_item.groups.keys()
    # for key, item in reviews_summaries:
    #     print(reviews_summaries.get_group(key))
    # reviews_summaries.get_group('A1YJEY40YUW4SE')[['reviewText', 'summary']]
