Recommendation System
=======

import nltk
nltk.download('punkt')

Install requirements
------------
pip install -r ./requirements.txt

Run dev server
------------
python manage_dev.py runserver 0.0.0.0:8000


Cron job
------------
`python manage_dev.py retrain_vedor_models` for retraining vendor models

`python manage_dev.py retrain_coupon_models` for retraining coupon models

[Cronjob_Django](https://blog.khophi.co/django-management-commands-via-cron/)

[Cronjob_Mac](https://ole.michelsen.dk/blog/schedule-jobs-with-crontab-on-mac-osx.html)

1. Open terminal and type: `env EDITOR=nano crontab -e`
2. A editor will open and type: (run retrain_model.sh and log to file retrain_model.log)

   `*/2 * * * * cd /User/teecoin/PycharmProjects/recommender_system_api/retrain_model.sh >> /User/teecoin/PycharmProjects/recommender_system_api/logs/retrain_model.log 2>&1`
   
3. Save it and wait

**Note**: Change your directory path in **retrain_model.sh** and **crontab** file

Tensorboard
----------------
Run `tensorboard --logdir ./logs`

Acknowledgements
----------------

- [Content-based Filtering](#)
- [Collaborative Filtering using Neural Network](https://arxiv.org/pdf/1708.05031.pdf)
- [Weighted Hybrid Recommendation](#)
- [Popularity Recommendation](#)
- [Recommendation using Implicit Feedback](#)

References
----------

[1] [Neural Collaborative Filtering](https://arxiv.org/pdf/1708.05031.pdf)  
[2] [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/pdf/1205.2618.pdf)  
[3] [Recommender System: Review](https://pdfs.semanticscholar.org/87d4/f4e19ad4fe140a40aebb24e4b7c6a9112332.pdf) 



