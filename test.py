from utils.score import score_user

local_loc = "/Users/anandmoghan/workspace/data/coswara/test_audio/"
final_score = score_user(local_loc=local_loc)
print(f"Final Score: {final_score}")
