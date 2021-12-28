import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def print_labels():

    print(
        "LABELS INFO:\n\n",
        "prob_min         : Minimum probability in a particular decile\n", 
        "prob_max         : Minimum probability in a particular decile\n",
        "prob_avg         : Average probability in a particular decile\n",
        "cnt_events       : Count of events in a particular decile\n",
        "cnt_resp         : Count of responders in a particular decile\n",
        "cnt_non_resp     : Count of non-responders in a particular decile\n",
        "cnt_resp_rndm    : Count of responders if events assigned randomly in a particular decile\n",
        "cnt_resp_wiz     : Count of best possible responders in a particular decile\n",
        "resp_rate        : Response Rate in a particular decile [(cnt_resp/cnt_cust)*100]\n",
        "cum_events       : Cumulative sum of events decile-wise \n",
        "cum_resp         : Cumulative sum of responders decile-wise \n",
        "cum_resp_wiz     : Cumulative sum of best possible responders decile-wise \n",
        "cum_non_resp     : Cumulative sum of non-responders decile-wise \n",
        "cum_events_pct   : Cumulative sum of percentages of events decile-wise \n",
        "cum_resp_pct     : Cumulative sum of percentages of responders decile-wise \n",
        "cum_resp_pct_wiz : Cumulative sum of percentages of best possible responders decile-wise \n",
        "cum_non_resp_pct : Cumulative sum of percentages of non-responders decile-wise \n",
        "KS               : KS Statistic decile-wise \n",
        "lift             : Cumuative Lift Value decile-wise",
         )



def decile_table(y_true, y_prob, change_deciles=10, labels=True, round_decimal=3):
    """Generates the Decile Table from labels and probabilities
    
    The Decile Table is creared by first sorting the customers by their predicted 
    probabilities, in decreasing order from highest (closest to one) to 
    lowest (closest to zero). Splitting the customers into equally sized segments, 
    we create groups containing the same numbers of customers, for example, 10 decile 
    groups each containing 10% of the customer base.
    
    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct/actual) target values.

        y_prob (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier/algorithm.

        change_deciles (int, optional): The number of partitions for creating the table
            can be changed. Defaults to '10' for deciles.

        labels (bool, optional): If True, prints a legend for the abbreviations of
            decile table column names. Defaults to True.

        round_decimal (int, optional): The decimal precision till which the result is 
            needed. Defaults to '3'.

    Returns:
        dt: The dataframe dt (decile-table) with the deciles and related information.

    Example:
        >>> import kds
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn import tree
        >>> X, y = load_iris(return_X_y=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=3)
        >>> clf = tree.DecisionTreeClassifier(max_depth=1,random_state=3)
        >>> clf = clf.fit(X_train, y_train)
        >>> y_prob = clf.predict_proba(X_test)
        >>> kds.metrics.decile_table(y_test, y_prob[:,1])
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    df = pd.DataFrame()
    df['y_true'] = y_true
    df['y_prob'] = y_prob
    # df['decile']=pd.qcut(df['y_prob'], 10, labels=list(np.arange(10,0,-1))) 
    # ValueError: Bin edges must be unique

    df.sort_values('y_prob', ascending=False, inplace=True)
    df['decile'] = np.linspace(1, change_deciles+1, len(df), False, dtype=int)

    # dt abbreviation for decile_table
    dt = df.groupby('decile').apply(lambda x: pd.Series([
        np.min(x['y_prob']),
        np.max(x['y_prob']),
        np.mean(x['y_prob']),
        np.size(x['y_prob']),
        np.sum(x['y_true']),
        np.size(x['y_true'][x['y_true'] == 0]),
    ],
        index=(["prob_min", "prob_max", "prob_avg",
                "cnt_cust", "cnt_resp", "cnt_non_resp"])
    )).reset_index()

    dt['prob_min']=dt['prob_min'].round(round_decimal)
    dt['prob_max']=dt['prob_max'].round(round_decimal)
    dt['prob_avg']=round(dt['prob_avg'],round_decimal)
    # dt=dt.sort_values(by='decile',ascending=False).reset_index(drop=True)

    tmp = df[['y_true']].sort_values('y_true', ascending=False)
    tmp['decile'] = np.linspace(1, change_deciles+1, len(tmp), False, dtype=int)

    dt['cnt_resp_rndm'] = np.sum(df['y_true']) / change_deciles
    dt['cnt_resp_wiz'] = tmp.groupby('decile', as_index=False)['y_true'].sum()['y_true']

    dt['resp_rate'] = round(dt['cnt_resp'] * 100 / dt['cnt_cust'], round_decimal)
    dt['cum_cust'] = np.cumsum(dt['cnt_cust'])
    dt['cum_resp'] = np.cumsum(dt['cnt_resp'])
    dt['cum_resp_wiz'] = np.cumsum(dt['cnt_resp_wiz'])
    dt['cum_non_resp'] = np.cumsum(dt['cnt_non_resp'])
    dt['cum_cust_pct'] = round(dt['cum_cust'] * 100 / np.sum(dt['cnt_cust']), round_decimal)
    dt['cum_resp_pct'] = round(dt['cum_resp'] * 100 / np.sum(dt['cnt_resp']), round_decimal)
    dt['cum_resp_pct_wiz'] = round(dt['cum_resp_wiz'] * 100 / np.sum(dt['cnt_resp_wiz']), round_decimal)
    dt['cum_non_resp_pct'] = round(
        dt['cum_non_resp'] * 100 / np.sum(dt['cnt_non_resp']), round_decimal)
    dt['KS'] = round(dt['cum_resp_pct'] - dt['cum_non_resp_pct'], round_decimal)
    dt['lift'] = round(dt['cum_resp_pct'] / dt['cum_cust_pct'], round_decimal)

    if labels is True:
        print_labels()

    return dt



def plot_lift(y_true, y_prob, title='Lift Plot', title_fontsize=14, 
              text_fontsize=10, figsize=None):
    """Generates the Decile based cumulative Lift Plot from labels and probabilities

    The lift curve is used to determine the effectiveness of a
    binary classifier. A detailed explanation can be found at
    http://www2.cs.uregina.ca/~dbd/cs831/notes/lift_chart/lift_chart.html
    The implementation here works only for binary classification.
    
    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_prob (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.

        title (string, optional): Title of the generated plot. Defaults to
            "Lift Plot".

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values (8, 10, 12, etc.)
            Defaults to 14.

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values (8, 10, 12, etc.)
            Defaults to 10.        

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

    Returns:
        None

    Example:
        >>> import kds
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn import tree
        >>> X, y = load_iris(return_X_y=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=3)
        >>> clf = tree.DecisionTreeClassifier(max_depth=1,random_state=3)
        >>> clf = clf.fit(X_train, y_train)
        >>> y_prob = clf.predict_proba(X_test)
        >>> kds.metrics.plot_lift(y_test, y_prob[:,1])
    """

    # Cumulative Lift Plot
    # plt.subplot(2, 2, 1)

    pl = decile_table(y_true,y_prob,labels=False)
    plt.plot(pl.decile.values, pl.lift.values, marker='o', label='Model')
    # plt.plot(list(np.arange(1,11)), np.ones(10), 'k--',marker='o')
    plt.plot([1, 10], [1, 1], 'k--', marker='o', label='Random')
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel('Deciles', fontsize=text_fontsize)
    plt.ylabel('Lift', fontsize=text_fontsize)
    plt.legend()
    plt.grid(True)
    # plt.show()



def plot_lift_decile_wise(y_true, y_prob, title='Decile-wise Lift Plot', 
                          title_fontsize=14, text_fontsize=10, figsize=None):
    """Generates the Decile-wise Lift Plot from labels and probabilities

    The lift curve is used to determine the effectiveness of a
    binary classifier. A detailed explanation can be found at
    http://www2.cs.uregina.ca/~dbd/cs831/notes/lift_chart/lift_chart.html
    The implementation here works only for binary classification.
    
    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_prob (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.

        title (string, optional): Title of the generated plot. Defaults to
            "Decile-wise Lift Plot".

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values (8, 10, 12, etc.)
            Defaults to 14.

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values (8, 10, 12, etc.)
            Defaults to 10.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

    Returns:
        None

    Example:
        >>> import kds
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn import tree
        >>> X, y = load_iris(return_X_y=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=3)
        >>> clf = tree.DecisionTreeClassifier(max_depth=1,random_state=3)
        >>> clf = clf.fit(X_train, y_train)
        >>> y_prob = clf.predict_proba(X_test)
        >>> kds.metrics.plot_lift_decile_wise(y_test, y_prob[:,1])
    """
    # Decile-wise Lift Plot
    # plt.subplot(2, 2, 2)
    pldw = decile_table(y_true,y_prob,labels=False)
    plt.plot(pldw.decile.values, pldw.cnt_resp.values / pldw.cnt_resp_rndm.values, marker='o', label='Model')
    # plt.plot(list(np.arange(1,11)), np.ones(10), 'k--',marker='o')
    plt.plot([1, 10], [1, 1], 'k--', marker='o', label='Random')
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel('Deciles', fontsize=text_fontsize)
    plt.ylabel('Lift @ Decile', fontsize=text_fontsize)
    plt.legend()
    plt.grid(True)
    # plt.show()



def plot_cumulative_gain(y_true, *y_scores, title='Cumulative Gain Plot',
                         title_fontsize=14, text_fontsize=10, figsize=None,
                         ax=None):
    """Generates the cumulative Gain Plot from labels and probabilities
    The cumulative gains chart is used to determine the effectiveness of a
    binary classifier. A detailed explanation can be found at
    http://www2.cs.uregina.ca/~dbd/cs831/notes/lift_chart/lift_chart.html 
    The implementation here works only for binary classification.
    
    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_prob (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.
        
        title (string, optional): Title of the generated plot. Defaults to
            "Decile-wise Lift Plot".
        
        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values (8, 10, 12, etc.)
            Defaults to 14.
        
        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values (8, 10, 12, etc.)
            Defaults to 10.
        
        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.
    
    Returns:
        None
    
    Example:
        >>> import kds
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn import tree
        >>> X, y = load_iris(return_X_y=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=3)
        >>> clf = tree.DecisionTreeClassifier(max_depth=1,random_state=3)
        >>> clf = clf.fit(X_train, y_train)
        >>> y_prob = clf.predict_proba(X_test)
        >>> kds.metrics.plot_cumulative_gain(y_test, y_prob[:,1])
    """

    # Cumulative Gains Plot
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot([0, 10], [0, 100], 'k--', marker='o', label='Random')
    
    for i, y_prob in enumerate(y_scores):
        pcg = decile_table(y_true,y_prob,labels=False)
        ax.plot(np.append(0, pcg.decile.values), np.append(0, pcg.cum_resp_pct.values), marker='o', label=f'Model {i+1}')
    
    ax.plot(np.append(0, pcg.decile.values), np.append(0, pcg.cum_resp_pct_wiz.values), 'c--', label=f'Wizard')
        
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel('Deciles', fontsize=text_fontsize)
    ax.set_ylabel('% Resonders', fontsize=text_fontsize)
    ax.legend()
    ax.grid(True)
    return ax



def plot_ks_statistic(y_true, y_prob, title='KS Statistic Plot', 
                      title_fontsize=14, text_fontsize=10, figsize=None):
    """Generates the KS Statistic Plot from labels and probabilities

    Kolmogorov-Smirnov (KS) statistic is used to measure how well the 
    binary classifier model separates the Responder class (Yes) from 
    Non-Responder class (No). The range of K-S statistic is between 0 and 1. 
    Higher the KS statistic value better the model in separating the 
    Responder class from Non-Responder class.
    
    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_prob (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.

        title (string, optional): Title of the generated plot. Defaults to
            "KS Statistic Plot".

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values (8, 10, 12, etc.)
            Defaults to 14.

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values (8, 10, 12, etc.)
            Defaults to 10.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

    Returns:
        None

    Example:
        >>> import kds
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn import tree
        >>> X, y = load_iris(return_X_y=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=3)
        >>> clf = tree.DecisionTreeClassifier(max_depth=1,random_state=3)
        >>> clf = clf.fit(X_train, y_train)
        >>> y_prob = clf.predict_proba(X_test)
        >>> kds.metrics.plot_ks_statistic(y_test, y_prob[:,1])
    """
    # KS Statistic Plot
    # plt.subplot(2, 2, 4)
    pks = decile_table(y_true, y_prob, labels=False)

    plt.plot(np.append(0, pks.decile.values), np.append(0, pks.cum_resp_pct.values),
             marker='o', label='Responders')
    plt.plot(np.append(0, pks.decile.values), np.append(0, pks.cum_non_resp_pct.values),
             marker='o', label='Non-Responders')
    # plt.plot(list(np.arange(1,11)), np.ones(10), 'k--',marker='o')
    ksmx = pks.KS.max()
    ksdcl = pks[pks.KS == ksmx].decile.values
    plt.plot([ksdcl, ksdcl],
             [pks[pks.KS == ksmx].cum_resp_pct.values,
              pks[pks.KS == ksmx].cum_non_resp_pct.values],
             'g--', marker='o', label='KS Statisic: ' + str(ksmx) + ' at decile ' + str(list(ksdcl)[0]))
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel('Deciles', fontsize=text_fontsize)
    plt.ylabel('% Resonders', fontsize=text_fontsize)
    plt.legend()
    plt.grid(True)
    # plt.show()



def report(y_true, y_prob, labels=True, plot_style = None, round_decimal=3, 
           title_fontsize=14, text_fontsize=10, figsize=(16, 10)):
    """Generates decile table and 4 plots (Lift, Lift@Decile, Gain and KS) 
    from labels and probabilities
    
    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_prob (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.

        labels (bool, optional): If True, prints a legend for the abbreviations of
            decile table column names. Defaults to True.

        plot_style(string, optional): Check available styles "plt.style.available".
            few examples: ['ggplot', 'seaborn', 'bmh', 'classic', 'dark_background', 
            'fivethirtyeight', 'grayscale', 'seaborn-bright', 'seaborn-colorblind', 
            'seaborn-dark', 'seaborn-dark-palette', 'tableau-colorblind10','fast'] 
            Defaults to ``None``.

        round_decimal (int, optional): The decimal precision till which the result is 
            needed. Defaults to '3'.

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values (8, 10, 12, etc.)
            Defaults to 14.

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values (8, 10, 12, etc.)
            Defaults to 10.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

    Returns:
        dc: The dataframe dc (decile-table) with the deciles and related information.

    Example:
        >>> import kds
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn import tree
        >>> X, y = load_iris(return_X_y=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=3)
        >>> clf = tree.DecisionTreeClassifier(max_depth=1,random_state=3)
        >>> clf = clf.fit(X_train, y_train)
        >>> y_prob = clf.predict_proba(X_test)
        >>> kds.metrics.report(y_test, y_prob[:,1])
    """
    
    dc = decile_table(y_true,y_prob,labels=labels,round_decimal=round_decimal)

    if plot_style is None:
        None
    else:
        plt.style.use(plot_style)
    
    fig = plt.figure(figsize=figsize)

    # Cumulative Lift Plot
    plt.subplot(2, 2, 1)
    plot_lift(y_true,y_prob)

    #  Decile-wise Lift Plot
    plt.subplot(2, 2, 2)
    plot_lift_decile_wise(y_true,y_prob)

    # Cumulative Gains Plot
    plt.subplot(2, 2, 3)
    plot_cumulative_gain(y_true,y_prob)

    # KS Statistic Plot
    plt.subplot(2, 2, 4)
    plot_ks_statistic(y_true,y_prob)

    return (dc)