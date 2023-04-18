# Building an Interactive Network Graph to Understand Communities

*A hands-on tutorial [scheduled](https://seattle2023.pydata.org/cfp/talk/83P9D7/) for [PyData Seattle 2023](https://pydata.org/seattle2023/schedule/) to build a [fun graph app](https://community-networks-pydata.uc.r.appspot.com/)!*

## Introduction -- People?!

**People are hard to understand, developers doubly so! In this tutorial, we will explore how communities form in organizations to develop a better solution than "The Org Chart". We will walk through using a few key Python libraries in the space, develop a toolkit for Clustering Attributed Graphs (more on that later) and build out an extensible interactive dashboard application that promises to take your legacy HR reporting structure to the next level.**

> In this tutorial, we will develop some fundamental knowledge on Graph Theory and capabilities in using key Python libraries to construct and analyze network graphs, including xarray, networkx, and dash-cytoscape. The goal of this talk is to build the tools you need to launch your own interactive dashboard in Python that can help explore communities of people based on shared characteristics (e.g. programming languages, projects worked on, apps used, management structure). The data we will dig into focuses on building a better understanding of developers + users and how they form communities, but this could just as easily be extended to any social network. The work we do here can be easily extended to your communities and use cases -- let's build something together!

> This talk is aimed at Pythonistas with beginner+ experience; we talk through some complex libraries and mathematical concepts, but beginners should be able to follow along and still build their understanding (and an app!)

* Follow along yourself in [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lucasdurand/network-graph-tutorial/HEAD)!
* **OR!** open me in [Codespaces](https://github.com/features/codespaces)

---

## Outline

### [Data](Part 0 - The Data/README.md)

* Generate representative sample data with [`Faker`](https://faker.readthedocs.io/en/master/), [`numpy`](https://numpy.org/doc/stable/), [`pandas`](https://pandas.pydata.org/docs/)

### [Building a *Simple* Network Graph](Part 1 - The Graph/README.md)

* Represent people as a network graph in [`networkx`](https://networkx.org/documentation/stable/index.html), visualize the graph with [`plotly`](https://plotly.com/python/)

### [Clustered Attribute Graphs](Part 2 - Clustered Graph Attributes/README.md)

* Introduce node attributes into the graph and explore clustering methods with [`networkx`](https://networkx.org/documentation/stable/index.html), [`node2vec`](https://github.com/eliorc/node2vec)
* Using *closeness* and *connectedness* to define communities and find peers with [`sklearn`](https://scikit-learn.org/stable/) and [`networkx.communities`](https://networkx.org/documentation/stable/index.html)

### [Exploring Communities with an Interactive App](Part 3 - An App/README.md)

* Build a an app with [`dash`](https://plotly.com/dash/) and [`dash-cytoscape`](https://dash.plotly.com/cytoscape) to expose our analytics toolkit and explore the communities inside our fictional company

## More Things?

* Deploy your app to *the Cloud!* This repo is set up to push to Google App Engine pretty easily, but give it a go with Heroku, Azure, or whatever the defacto (semi)-free option is these days!
* Build better sample data, try different types of companies, communities, etc. and look for useful patterns
* Add a feature to show how two people are connected (choose two nodes and then draw some/all of the paths between them)
* Visualize larger networks -- our system starts to break down over ~500 nodes and needs pagination + sharding (?)

## Deployment

### Google App Engine

1. Install the Google Cloud SDK https://cloud.google.com/sdk/docs/install-sdk#deb
2. Initialize the repo `gcloud init` and log in. Create a new project if needed.
2. Configure your `app.yaml` https://cloud.google.com/appengine/docs/standard/reference/app-yaml?tab=python
1. Deploy the app `gcloud app deploy`
1. View it! `gcloud app browse` or -- https://community-networks-pydata.uc.r.appspot.com/
