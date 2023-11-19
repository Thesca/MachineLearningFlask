from website import app
from flask import render_template, request

@app.route("/", methods=['GET', 'POST'])
def homepage():
    if request.method == 'POST':
        clf_escolhido = request.form['clf_selected']
        if clf_escolhido == "1":
            clf_escolhido == "DTC"
        elif clf_escolhido == "2":
            clf_escolhido == "MLP"
        elif clf_escolhido == "3":
            clf_escolhido == "KNN"
        elif clf_escolhido == "4":
            clf_escolhido == "RFC"
        return render_template("index.html", clf = clf_escolhido)
    else:
        return render_template("index.html")
    
@app.route("/treinar/<int:clf_selected>", methods=['GET', 'POST'])
def treinamento(clf_selected):
    from matplotlib import pyplot as plt
    from sklearn.datasets import load_iris
    from sklearn import tree
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier

    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    if request.method == 'POST':
        print(request.form)
        var1 =  request.form['var1']
        var2 =  request.form['var2']
        var3 =  request.form['var3']
        if(clf_selected == 1):
            dtClf = DecisionTreeClassifier(max_depth=var1, random_state=var2, max_leaf_nodes=var3)
            dtClf.fit(X_train, y_train)
            clfAcc = dtClf.score(X_test, y_test)
            y_pred = dtClf.predict(X_test)
        elif(clf_selected == 2):
            mlpClf = MLPClassifier(hidden_layer_sizes=var1, random_state=var2, learning_rate=var3)
            mlpClf.fit(X_train, y_train)
            clfAcc = mlpClf.score(X_test, y_test)
            y_pred = mlpClf.predict(X_test)
        elif(clf_selected == 3):
            knnClf = KNeighborsClassifier(n_neighbors=var1, leaf_size=var2, power_parameter=var3)
            knnClf.fit(X_train, y_train)
            clfAcc = knnClf.score(X_test, y_test)
            y_pred = knnClf.predict(X_test)
        elif(clf_selected == 4):  
            randfClf = RandomForestClassifier(n_estimators=var1, max_depth=var2, max_leaf_nodes =var3)
            randfClf.fit(X_train, y_train)
            clfAcc = randfClf.score(X_test, y_test)
            y_pred = randfClf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        classes = iris.target_names.tolist()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot()
        plt.show()
        macroAvg = f1_score(y_test, y_pred, average='macro')
        return render_template('index.html', acc=clfAcc, macro=macroAvg)
    else:
        return render_template('index.html')

