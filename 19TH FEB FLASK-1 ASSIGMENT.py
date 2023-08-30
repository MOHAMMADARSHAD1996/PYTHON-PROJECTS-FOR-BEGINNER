#!/usr/bin/env python
# coding: utf-8

# #  <P style="color:brown"> MD. ARSHAD </p>
# ###  <P style="color:skyblue"> ALISHARMEEN02@GMAIL.COM  </p>

# Q1. What is Flask Framework? What are the advantages of Flask Framework?

Flask is a micro web framework for Python. It is designed to be simple and lightweight, allowing developers to quickly build web applications with minimal overhead. Flask provides the essentials needed to create web applications, while giving developers the flexibility to choose and integrate additional tools and libraries as needed.

Advantages of Flask Framework:

Simplicity: Flask is known for its simplicity and minimalistic design. It provides just the basics needed to build web applications, making it easy for developers to understand and work with.

Flexibility: Flask doesn't impose a specific structure or architecture on developers. This gives developers the freedom to structure their projects according to their preferences and needs.

Modularity: Flask follows a modular design, allowing developers to pick and choose components (extensions) based on the requirements of their application. This modularity helps keep the codebase clean and efficient.

Extensibility: Flask has a wide range of extensions available that provide additional functionality such as database integration, authentication, form handling, and more. Developers can add these extensions to their projects as needed.

Jinja2 Templating: Flask uses the Jinja2 templating engine, which allows developers to separate HTML and dynamic content, making it easier to manage and render dynamic data within HTML templates.

Built-in Development Server: Flask comes with a built-in development server, making it simple to test and debug applications locally before deploying them to production servers.

Werkzeug Integration: Flask is built on top of the Werkzeug WSGI (Web Server Gateway Interface) toolkit, which provides low-level components for handling HTTP requests and responses. This integration aids in handling routing, URL generation, and more.

Community and Documentation: Flask has an active and supportive community of developers, which means you can find a wealth of resources, tutorials, and documentation to assist you in your development journey.

RESTful API Support: While Flask doesn't enforce a specific architecture, it's well-suited for building RESTful APIs. It provides tools and libraries to easily handle HTTP methods, request parsing, and response formatting.

Minimal Overhead: Flask's lightweight design and minimal overhead make it a great choice for small to medium-sized applications where heavy frameworks might be overkill.

Learning Curve: Due to its simplicity, Flask can be quickly learned by developers new to web development or Python.

Pythonic: Flask adheres to Python's design principles, making it intuitive for Python developers to work with.

In summary, Flask is a micro web framework that offers simplicity, flexibility, and modularity. It's an excellent choice for developers who want to build web applications quickly without being bound by a rigid framework structure.
# In[ ]:





# Q2. Create a simple Flask application to display ‘Hello World!!’. Attach the screenshot of the output in
# Jupyter Notebook.
Sure, here's a simple example of a Flask application that displays "Hello World!!" when you access the root URL:
from flask import Flask

# Create a Flask application instance
app = Flask(__name__)

# Define a route for the root URL
@app.route('/')
def hello_world():
    return 'Hello World!!'

# Run the application if this script is executed
if __name__ == '__main__':
    app.run()
To run this application, follow these steps:

Make sure you have Flask installed. You can install it using the following command:

pip install Flask
Save the above code in a file named app.py or any other suitable name.

Open a terminal and navigate to the directory where the app.py file is located.

Run the Flask application:
The application will start and you'll see output similar 
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
Open a web browser and go to http://127.0.0.1:5000/ or http://localhost:5000/. You should see the "Hello World!!" message displayed in your browser.

To stop the Flask application, press CTRL+C in the terminal.

This simple example demonstrates how to create a basic Flask application and define a route to handle the root URL. When you access the root URL, the hello_world() function is executed, and it returns the "Hello World!!" message.
# In[ ]:





# Q3. What is App routing in Flask? Why do we use app routes?
# In Flask, app routing refers to the process of mapping URLs (Uniform Resource Locators) to specific functions or views within your web application. Each URL corresponds to a specific route, and when a user accesses that URL, the associated function is executed, generating an appropriate response.

Flask uses the @app.route() decorator to define routes within your application. This decorator specifies the URL pattern that should trigger the execution of a particular function. Routes are an essential aspect of web frameworks like Flask as they determine how different URLs are handled and what content is displayed to users.

Here's an example of using app routing in Flask:
from flask import Flask

app = Flask(__name__)

# Define a route for the root URL
@app.route('/')
def hello_world():
    return 'Hello World!!'

# Define a route for another URL
@app.route('/about')
def about():
    return 'This is the About page.'

if __name__ == '__main__':
    app.run()
In this example, the @app.route() decorator is used to define two routes. The first route corresponds to the root URL ("/"), and when a user accesses this URL, the hello_world() function is executed, returning "Hello World!!". The second route ("/about") triggers the about() function, which returns "This is the About page."

Why Do We Use App Routes in Flask?

URL Mapping: App routes provide a clean and organized way to map URLs to specific functions or views in your application. This makes it easy to understand how different URLs are handled and what content is displayed to users.

Logical Separation: App routes allow you to logically separate different sections or functionalities of your application. This promotes a modular structure and makes your codebase more manageable.

Dynamic Content: App routes enable you to create dynamic web pages by passing variables within the URL. These variables can be captured by the route function and used to generate customized content.

RESTful APIs: App routes are essential for building RESTful APIs. Each route can correspond to a specific API endpoint, and different HTTP methods (GET, POST, PUT, DELETE) can trigger different functions for data retrieval, creation, updating, and deletion.

User Interaction: Routes allow you to define how users interact with different parts of your application. For example, you can define routes for user registration, login, profile viewing, and more.

Clean URLs: Well-defined routes result in clean and meaningful URLs, which can improve user experience and search engine optimization.

Separation of Concerns: App routes promote the separation of concerns by ensuring that each route function handles a specific task or piece of functionality. This makes your code more modular and maintainable.

In summary, app routing in Flask is the mechanism by which URLs are mapped to specific functions or views within your application. App routes provide organization, modularity, and dynamic content generation, making your Flask application more structured and user-friendly.
# In[ ]:





# Q4. Create a “/welcome” route to display the welcome message “Welcome to ABC Corporation” and a “/”
# route to show the following details:
# Company Name: ABC Corporation
# Location: India
# Contact Detail: 999-999-9999
# Attach the screenshot of the output in Jupyter Notebook.
# Certainly! Here's an example of how you can create the desired routes in a Flask application:
from flask import Flask

app = Flask(__name__)

# Define a route for the /welcome URL
@app.route('/welcome')
def welcome():
    return 'Welcome to ABC Corporation'

# Define a route for the root URL
@app.route('/')
def company_details():
    details = """
    Company Name: ABC Corporation
    Location: India
    Contact Detail: 999-999-9999
    """
    return details

if __name__ == '__main__':
    app.run()
In this example, the /welcome route is defined using the @app.route('/welcome') decorator. When a user accesses the /welcome URL, the welcome() function is executed, and it returns the "Welcome to ABC Corporation" message.

The root route (/) is defined using the @app.route('/') decorator. When a user accesses the root URL, the company_details() function is executed, and it returns the company details formatted as a multi-line string.

To run this Flask application, follow the same steps as mentioned earlier. After running the application, you can access the /welcome and root URLs in your web browser or through a tool like curl to see the messages displayed on those routes.
# In[ ]:





# Q5. What function is used in Flask for URL Building? Write a Python code to demonstrate the working of the
# url_for() function.
In Flask, the url_for() function is used for URL building. It generates a URL for a given view function, taking into account the routing rules defined in your application. This function helps you avoid hardcoding URLs in your templates and code, making your application more maintainable and adaptable.

Here's an example of how to use the url_for() function in Flask:
from flask import Flask, url_for

app = Flask(__name__)

@app.route('/')
def index():
    return 'This is the index page.'

@app.route('/about')
def about():
    return 'This is the About page.'

@app.route('/contact')
def contact():
    return 'This is the Contact page.'

if __name__ == '__main__':
    with app.test_request_context():
        # Using url_for() to generate URLs for different routes
        index_url = url_for('index')
        about_url = url_for('about')
        contact_url = url_for('contact')

        print(f'URL for index: {index_url}')
        print(f'URL for about: {about_url}')
        print(f'URL for contact: {contact_url}')
In this example, the url_for() function is used within a with app.test_request_context(): block. This is necessary to set up a temporary request context, which allows you to use the url_for() function outside of an actual request.

The url_for() function takes the name of a view function as an argument and generates the corresponding URL based on the routing rules you've defined in your application. The generated URLs are printed to the console.

When you run this code, you should see the URLs for the index, about, and contact routes printed to the console. The URLs will be dynamically generated based on the routes you've defined, making it easy to update and maintain your application's URLs.
# #  <P style="color:GREEN"> Thank You ,That's All </p>
