from pandas import concat, DataFrame, read_csv

# df_transcript = read_csv("../data/transcipt.csv").drop(["conversation_id","comment_number"], axis=1)
# df_reddit = read_csv("../data/casual_data_windows.csv", index_col=0).drop(["2"], axis=1)
# df_reddit.columns = ["comment", "response"]
df = DataFrame(columns=["comment", "response"])
df_ai = read_csv("data/chatterbot_csv/ai.csv")
df_bot = read_csv("data/chatterbot_csv/botprofile.csv")
df_computers = read_csv("data/chatterbot_csv/computers.csv")
df_emotion =  read_csv("data/chatterbot_csv/emotion.csv")
df_food = read_csv("data/chatterbot_csv/food.csv")
df_gossip = read_csv("data/chatterbot_csv/gossip.csv")
df_greetings = read_csv("data/chatterbot_csv/greetings.csv")
df_health = read_csv("data/chatterbot_csv/health.csv")
df_humor = read_csv("data/chatterbot_csv/humor.csv")
df_literature = read_csv("data/chatterbot_csv/literature.csv")
df_money = read_csv("data/chatterbot_csv/money.csv")
df_movies = read_csv("data/chatterbot_csv/movies.csv")
df_politics = read_csv("data/chatterbot_csv/politics.csv")
df_psych = read_csv("data/chatterbot_csv/psychology.csv")
df_science = read_csv("data/chatterbot_csv/science.csv")
df_sports = read_csv("data/chatterbot_csv/sports.csv")
df_trivia = read_csv("data/chatterbot_csv/trivia.csv")
# df = concat([df_transcript, df_reddit, df_movies, df_ai, df_bot, df_computers,
#              df_emotion, df_food, df_gossip, df_greetings, df_health, df_humor,
#              df_literature, df_money, df_politics, df_psych, df_science,
#              df_sports, df_trivia], join="outer", axis=0)
df = concat([df_movies, df_ai, df_bot, df_computers,
             df_emotion, df_food, df_gossip, df_greetings, df_health, df_humor,
             df_literature, df_money, df_politics, df_psych, df_science,
             df_sports, df_trivia], join="outer", axis=0)
df = df.dropna(axis=0)
df = df.drop_duplicates()
df.to_csv("data/master_data.csv", index=False)

print(len(df))
