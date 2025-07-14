#Image upload and all
CLOUDINARY_URL=cloudinary://649548768487497:U-4VuN6r5yVguM-HojrNkW274_U@dwujrwjee

# DB
MONGODB_URI=mongodb+srv://premo625geet:mFPYNVI7laNszKsh@mentaid-cluster.stpvkxg.mongodb.net/?retryWrites=true&w=majority&appName=mentaid-cluster

#Auth +encyrption
JWT_SECRET_KEY=TLdgMUaDohPgZHqbmj8o8bT--RhHsGZjNd2XOsILqX4ZxPs8EvDPcDzu5kH362Ci-_Il-M5CauyQ53v6pnXPaw
ENCRYPTION_KEY= "N-oiW-K73Ur0fM6yCwDiUxLev_7fCwn2FKUsEF_8Wt4="

#LLMS AND summarizer
OPENROUTER_API_KEY=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

#For whisper and possibly others
GROQ_API_KEY=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



above are the env vairbvales that are saved in the .env file in the root directory.


now what i want you to do is ..make me the prototype using streamlit and python as backend.

what i simply want is two interfaces

one for patient/common user interface where the user has the option to give the mood of the day on a scale of 5 emojies and the patient/user should be able to write the journal entry and thats it.


and for the clinician dashboard i want to see the prediction given by the svm model and the nlp transformer model mentalBERT

the shap and lime of both of the model properly


the steps i want you to follow is 
1)Lay out the requirements of everything you need to make this prototype .
2)Go through the notebooks named NLP_TRANSFORMER_1_2 and SVM 1_2 in notebooks folder and know exactly howthey workd and everything and etc.
2.5)Go through the MODELS MADE FOLDER TOO TO SEE WHAT WE HAVE THE MODELS
3)Share your findings and whether you know everything to actually implement the prototype
4)Lay out the plan and exact steps you are gonna take to implement the prototype.
5)After that if i allow , i want you to start implementing the prototype.


Also just so you know i want you to implement the prototype in the prototype folder inside the backend folder.

also just so you know you will need to incorparate chunking pipleline where the input journal entry gets chunksed into overlapping chunks (avg length 931 for svm and 512 tokens for nlp transformer model),gets processed and then the avg or whatever is taken and show for each model.also give an ensemble output depening on the output of both the model.


ALL THE BEST!!!