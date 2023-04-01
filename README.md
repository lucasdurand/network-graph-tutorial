# peopleanalytics
Sample code to build visual analytics of how people connect and appreciate one another in an organisation

Try it out yourself in [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lucasdurand/peopleanalytics/HEAD)

## Deployment

### Google App Engine

1. Install the Google Cloud SDK https://cloud.google.com/sdk/docs/install-sdk#deb
2. Initialize the repo `gcloud init` and log in. Create a new project if needed.
2. Configure your `app.yaml` https://cloud.google.com/appengine/docs/standard/reference/app-yaml?tab=python
1. Deploy the app `gcloud app deploy`
1. View it! `gcloud app browse` or -- https://community-networks-pydata.uc.r.appspot.com/