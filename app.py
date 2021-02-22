#Importing libraries
from flask import Flask, request
from flask import * 
from flask import Flask, render_template, request, json, Response,redirect,flash,url_for,session,jsonify
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
import helper as helper
import segmentation as segmentation
import base64
from werkzeug.utils import secure_filename 
import os
import uuid


#app
app = Flask(__name__,static_url_path='',static_folder='web/static',
			template_folder='web/templates')
			
#app config			
app.secret_key = "EnterpriseAI" 
app.config["MONGO_URI"] = "mongodb://<account-name>:<password>@cluster0-shard-00-00.zwnev.mongodb.net:27017,cluster0-shard-00-01.zwnev.mongodb.net:27017,cluster0-shard-00-02.zwnev.mongodb.net:27017/<project-name>?ssl=true&replicaSet=atlas-bevp92-shard-0&authSource=admin&retryWrites=true&w=majority"
app.config["BASEPATH"] = 'D:\ENTERPRISE_AI\env\datas'

#mongodb init
mongodb_client = PyMongo(app)
db = mongodb_client.db 
print("DB Connected")  

#home 
@app.route('/',methods=['GET','POST'])
def home():    
	if 'username' in session:
		print('notlogeed in yet')
		return render_template('index.html',message={"msg":'logedin',"session":session})	
	return render_template('index.html',message={"msg":'notloged',"session":session})	

#login
@app.route('/login',methods=['GET','POST'])
def login():
	# session
	if 'username' not in session:
		if request.method == 'POST':
			email = request.form['email']
			password = request.form['password']
			try:
				User = db.User.find_one({"email": email})
				if User:
					#Authentication
					password_check = check_password_hash(User["password"],password)					                        
					if password_check:
						#if user-> role = admin 						
						if User["role"] == "admin":							
							session['username']=User["name"]      
							session['email']=User["email"]
							session['role']="admin"
							return redirect(url_for('ecommerce'))

						else:							  
							session['username']=User["name"]      
							session['email']=User["email"]
							session['role']="customer"                         

						return render_template('login.html',message={"msg":'Login Success',"session":session})
					else: 
						return render_template('login.html',message={"msg":'Please Check your Credentials',"session":session})
				else:
					return render_template('login.html',message={"msg":'Please Check your Credentials'}) 

			except Exception as e:
				print(e)
				return render_template('login.html',message={"msg":'Please Check your Credentials'})
	else:
		return render_template('login.html',message={"msg":'logedin',"session":session}) 

	return render_template('login.html',message='')

#signup
@app.route('/create',methods=['GET','POST'])
def createaccount():
	if request.method == 'POST':
		try:
			# print(request.form) 
			name = request.form['name']
			email = request.form['email']
			password = request.form['password']
			gender = request.form['Gender']
			dob = request.form['dob']
			state = request.form['State']
			country = request.form['Country']
			password_hash = generate_password_hash(password)
			age = helper.calculateAge(str(dob))
			address = {"state":state,"country":country}
			#create user with details
			db.User.insert_one({'name':name,'email': email, 'password': password_hash,'gender':gender,"dob": str(dob),"age":age,"address":address,"role":"customer"}) 
			# print('User Account Created')
			return render_template('createaccount.html',message='Account Created Successfully')
 
		except Exception as e:  
			print(e)
			print("could not Create account")
			return render_template('createaccount.html',message='Couldnt create account')  
	return render_template('createaccount.html')


#logout
@app.route('/logout',methods=['GET','POST'])
def logout():
	try: 
		session.pop('username', None)
		session.pop('email', None) 
		session.pop('role', None) 
		print(session)
		return redirect(url_for('home'))
	except:
		return redirect(url_for('home'))


#product
@app.route('/product',methods=['GET','POST'])
def product():
	prdsDetails = [] 
	try:
		Products = db.products.find()	
		for x in Products:
			prd={}
			print(x["productID"])
			prd["productid"] = x["productID"]
			prd["productname"] = x["productname"] 
			prd["productdescription"] = x["productdescription"] 
			prd["price"] = x["price"] 
			prd["stock"] = x["stock"]	
			decodeImg = x["image"].decode()
			prd["imgsrc"] =  decodeImg
			prdsDetails.append(prd)

		if 'username' in session:
			return render_template('product.html',message={"msg":'logedin',"session":session,"prds":prdsDetails})

		return render_template('product.html',message={"prds":prdsDetails})		

	except Exception as ae:
		print(ae)
		return render_template('product.html',message={"prds":prdsDetails})
	return render_template('product-detail.html',message={"msg":'',"session":session,"prds":prdsDetails})

#product Details
@app.route('/details',methods=['GET','POST'])
def product_details():
	prdid = request.args.get('prdid')
	print(prdid)	
	try:
		Products = db.products.find_one({"productID": prdid})
		print(Products)			
		prd={}	 
		prd["productname"] = Products["productname"] 
		prd["productdescription"] = Products["productdescription"] 
		prd["price"] = Products["price"] 
		prd["stock"] = Products["stock"]	
		decodeImg=Products["image"].decode() 
		prd["imgsrc"] =  decodeImg

		if 'username' in session:
			return render_template('product-detail.html',message={"msg":'logedin',"session":session,"prds":prd})
		
		return render_template('product-detail.html',message={"msg":'',"session":session,"prds":prd})

	except Exception as ae:
		print(ae)
	return render_template('product-detail.html',message={"msg":'logedin',"session":session,"prds":prd})
		

#about
@app.route('/about',methods=['GET','POST'])
def about():
	if 'username' in session:
		return render_template('about.html',message={"msg":'logedin',"session":session})
	return render_template('about.html',message='')

#blog
@app.route('/blog',methods=['GET','POST'])
def blog():
	return render_template('blog.html')  

#cart
@app.route('/cart',methods=['GET','POST'])
def cart():    
	if 'username' in session:
		return render_template('cart.html',message={"msg":'logedin',"session":session}) 
	return render_template('cart.html',message='')

#contact
@app.route('/contact',methods=['GET','POST'])
def contact():
	if 'username' in session:
		return render_template('contact.html',message={"msg":'logedin',"session":session})
	return render_template('contact.html',message='')

#admin - create product 
@app.route('/admin/createproduct',methods=['GET','POST'])
def product_create():
	if session['role'] == "admin":
		if request.method == 'POST':
			productname = request.form['productname'] 
			productdescription = request.form['productdescription']
			price = request.form['price']
			stock = request.form['stock']		
			file = request.files['Imagefile']
			try:
				filename = secure_filename(file.filename)                
				file.save(os.path.join(app.config['BASEPATH'], filename))
				uploade_img_path =  os.path.join(app.config['BASEPATH'], filename) 
				with open(uploade_img_path, "rb") as image_file:
					encoded_string = base64.b64encode(image_file.read())				   
					productID=str(uuid.uuid4())
					productsDetails = {"productID":productID,"productname":productname,"productdescription":productdescription,"price":int(price),"stock":int(stock),"image":encoded_string}    				
					res=db.products.insert(productsDetails)
					print(productsDetails)
				return jsonify({"message":"FileUploadedSucessfull","productID":productID,"session":session}) 

			except Exception as ae:
				print(ae)
				return jsonify({"message":"Filenotuploaded","productID":'error generating product ID',"session":session}) 
	else:
		return 'access denied'
	return render_template('productcreate.html',message={"msg":'',"session":session})
	
#admin - ecommerce
@app.route('/admin/ecommerce',methods=['GET','POST'])
def ecommerce():	
	if session['role'] == "admin":
		User = db.User.find()
		prdsDetails = [] 
		try:
			Products = db.products.find()	
			for x in Products:
				prds={}
				print(x["productID"])
				prds["productid"] = x["productID"]
				prds["productname"] = x["productname"] 
				prds["productdescription"] = x["productdescription"] 
				prds["price"] = x["price"] 
				prds["stock"] = x["stock"]	
				decodeImg = x["image"].decode()
				prds["imgsrc"] =  decodeImg
				prdsDetails.append(prds)			
			analytics = segmentation.segmentation()
			print("vals: ",analytics)			
			return render_template('adminecommerce.html',message={"msg":'logedin',"session":session,"analytics":analytics,"users":User,"products":prdsDetails})	

		except Exception as ae:
			print(ae)
			return render_template('adminecommerce.html',message={"msg":'',"session":session,"analytics":''})
	else:
		print('access denied')
		return redirect(url_for("home"))
	return render_template('adminecommerce.html',message={"msg":'',"analytics":''})
	
#admin - analytics 
@app.route('/admin/analytics',methods=['GET','POST'])
def analytics():
	if session['role'] == "admin":	
		return render_template('adminanalytics.html',message={"msg":'logedin',"session":session})
	else:
		print('access denied')
		return redirect(url_for("home"))

#admin - productview
@app.route('/admin/productsview',methods=['GET','POST'])
def productsview():
	try:
		if session['role'] == "admin":
			prdsDetails = []
			Products = db.products.find()
			prdcount=0	
			for x in Products:
				prd={}
				print(x["productID"])
				prdcount += 1
				prd["prdcount"] = prdcount
				prd["productid"] = x["productID"]
				prd["productname"] = x["productname"] 
				prd["productdescription"] = x["productdescription"] 
				prd["price"] = x["price"] 
				prd["stock"] = x["stock"]	
				decodeImg = x["image"].decode()
				prd["imgsrc"] =  decodeImg
				prdsDetails.append(prd)
			return render_template('adminproductview.html',message={"msg":'logedin',"session":session,"prds":prdsDetails})
		else:
			print('access denied')
			return redirect(url_for("home"))

	except Exception as ae:
		print(ae)
		return render_template('adminproductview.html',message={"msg":'',"session":session})		
	return redirect(url_for('home')) 




#start app 
if __name__ == '__main__':
	app.run(debug=True) 


