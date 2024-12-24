from dataloaders.base import Dataset

class EQDataset(Dataset):
    def __init__(self, dataset_name=None, dataset_file_path=None, rank=None, world_size=None, image_url=None, preloaded_image_num=1):
        super().__init__(dataset_name=dataset_name, dataset_file_path=dataset_file_path, rank=rank, world_size=world_size, image_url=image_url, preloaded_image_num=preloaded_image_num)

    def calculate(self, data, base_dict, base_calculate_kwargs):

        import math
        gold = eval(base_calculate_kwargs['gold'])

        try:
            filtered_r = dict(base_calculate_kwargs['filtered_r'])
        except Exception as e:
            return {"score": 0, "percent_parseable": 0}

        if len(filtered_r.items()) != 4:
            return {"score": 0, "percent_parseable": 0}
        emotions_dict = {}
        for emotion, user_emotion_score in filtered_r.items():
            for i in range(1, 5):
                if emotion == gold[f"emotion{i}"]:
                    emotions_dict[emotion] = True
        if len(emotions_dict) != 4:
            print("! Error: emotions did not match reference")
            print(filtered_r)
            return {"score": 0, "percent_parseable": 0}

        difference_tally = (
            0  # Tally of differerence from reference answers for this question
        )

        # Iterate over each emotion in the user's answers.
        for emotion, user_emotion_score in filtered_r.items():
            # If this emotion is in the reference, calculate the difference between the user's score and the reference score.
            for i in range(1, 5):
                if emotion == gold[f"emotion{i}"]:
                    d = abs(
                        float(user_emotion_score) - float(gold[f"emotion{i}_score"])
                    )
                    # this will be a value between 0 and 10
                    if d == 0:
                        scaled_difference = 0
                    elif d <= 5:
                        # S-shaped scaling function
                        # https://www.desmos.com/calculator
                        # 6.5\cdot\ \frac{1}{\left(1\ +\ e^{\left(-1.2\cdot\left(x-4\right)\right)}\right)}
                        scaled_difference = 6.5 * (1 / (1 + math.e ** (-1.2 * (d - 4))))

                    else:
                        scaled_difference = d
                    difference_tally += scaled_difference

        # Inverting the difference tally so that the closer the answer is to reference, the higher the score.
        # The adjustment constant is chosen such that answering randomly produces a score of zero.
        adjust_const = 0.7477
        final_score = 10 - (difference_tally * adjust_const)
        final_score_percent = final_score * 10

        return {"score": final_score_percent, "percent_parseable": 100}

data_core = EQDataset