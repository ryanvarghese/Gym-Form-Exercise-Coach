
  // Your web app's Firebase configuration
  // For Firebase JS SDK v7.20.0 and later, measurementId is optional
  var firebaseConfig = {
    apiKey: "AIzaSyAj5DiwbzX7jXW57xTaeiWPWkCZXOhbT5Q",
  authDomain: "minip-1ae85.firebaseapp.com",
  projectId: "minip-1ae85",
  storageBucket: "minip-1ae85.appspot.com",
  messagingSenderId: "319175939377",
  appId: "1:319175939377:web:064d6dafd63ce643c54c10",
  measurementId: "G-L2SE4D7YF7"
  };
  // Initialize Firebase
  firebase.initializeApp(firebaseConfig);

  const auth =  firebase.auth();

  //signup function
  function signUp(){
    var email = document.getElementById("email");
    var password = document.getElementById("password");

    const promise = auth.createUserWithEmailAndPassword(email.value,password.value);
    //
    promise.catch(e=>alert(e.message));
    alert("SignUp Successful");
    window.location.replace('homepage.html');
  }

  //signIN function
  function  signIn(){
    var email = document.getElementById("email");
    var password  = document.getElementById("password");
    const promise = auth.signInWithEmailAndPassword(email.value,password.value);
    promise.catch(e=>alert(e.message));
    window.location.replace('homepage.html');
    
  }


  //signOut

  function signOut(){
    auth.signOut();
    alert("SignOut Successfully from System");
    window.location.replace('login.html');
  }

  //active user to homepage
  firebase.auth().onAuthStateChanged((user)=>{
    if(user){
      var email = user.email;
      alert("Active user "+email);

    }else{
      // alert("No Active user Found")
    }
  })