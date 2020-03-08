from pandas import DataFrame

df = DataFrame(columns=["conversation_id", "comment_number", "comment", "response"]);
with open("transcript", "r") as infile:
    num_lines = 0
    
    conversation_id = 0
    comment_num = 0
    comment = ""
    response = ""

    for line in infile:

        # if its a new line, its a new conversation
        if line == "\n":
            conversation_id += 1
            comment_num = 0
            commenet = ""
            response = ""
        else:
            comment_num += 1

            # save the previous response as the new comment
            commenet = response

            # get the response
            # some have two comments from the same speaker that go onto a new line
            if len(line.split(": ")) == 1:
                response = line
            else:
                response = line.split(": ")[1].strip(" \n")

            # don't save something that doesn't have a response
            if comment_num > 1:
                num_lines += 1
                list_data = [conversation_id, comment_num, commenet, response]
                df.loc[num_lines] = list_data

print(df.head())
print("number of rows:", len(df))

df.to_csv("data/transcipt.csv",index=False)


        
