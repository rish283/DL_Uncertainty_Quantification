import numpy as np


class perf:

    def roc(model, x_test, yts, sm1, sm2, sm4, sm5, sm7, sm8):

        pmt = model.predict(x_test)

        prt = np.zeros((len(pmt), 10))
        pr1 = np.zeros(len(pmt))

        aa1 = np.max(pmt, 1)
        for i in range(len(aa1)):
            for j in range(10):
                if pmt[i, j] == aa1[i]:
                    prt[i, j] = 1
                else:
                    prt[i, j] = 0

        for i in range(len(x_test)):
            for j in range(10):
                if int(prt[i, j]) == 1:
                    pr1[i] = j

        cre = np.zeros(len(pmt))
        cre1 = np.zeros(len(pmt))
        ct = 0
        for i in range(len(pmt)):
            if int(pr1[i]) == yts[i]:
                cre[i] = 1
                cre1[i] = 0

                ct += 1
            else:
                cre[i] = 0
                cre1[i] = 1

        from sklearn import metrics
        import matplotlib.pyplot as plt
        from numpy import cov
        from scipy.stats import pearsonr as per
        from scipy.stats import spearmanr as sper
        from scipy.stats import pointbiserialr as pbs

        L = cre1
        L = L.astype(bool)

        fpr1, tpr1, _ = metrics.roc_curve(cre1, sm1)
        auc1 = metrics.roc_auc_score(cre1, sm1)
        f11 = metrics.precision_recall_curve(cre1, sm1)
        f11 = [f11[0][0:-1], f11[1][0:-1], f11[2][0:-1]]
        puc1 = metrics.auc(f11[1], f11[0])
        fs1 = metrics.average_precision_score(cre1, sm1)
        cv1 = cov(cre1, sm1)[0][1]
        prs1, _ = per(cre1, sm1)
        sprs1, _ = sper(cre1, sm1)
        pb1, _ = pbs(cre1, (sm1 - np.min(sm1))/(np.max(sm1) - np.min(sm1)))
        sa1 = (sm1 - np.min(sm1))/(np.max(sm1) - np.min(sm1))
        # cv1 = cov(cre1, sa1)[0][1]

        h11, h12 = sm1[L], sm1[~L]
        hn = np.concatenate([h11, h12])
        hn = (hn - np.min(hn))/(np.max(hn) - np.min(hn))
        h11 = hn[0:len(h11)]
        h12 = hn[len(h11): len(h12)]
        md1 = abs(np.mean(h11) - np.mean(h12))

        sd = f11
        f1s1 = 0
        for u in range(len(sd[0])):
            f1s1 += 2*sd[0][u]*sd[1][u]/(sd[0][u] + sd[1][u])
        f1s1 /= len(sd[0])

        fpr2, tpr2, _ = metrics.roc_curve(cre1, sm2)
        auc2 = metrics.roc_auc_score(cre1, sm2)
        f12 = metrics.precision_recall_curve(cre1, sm2)
        f12 = [f12[0][0:-1], f12[1][0:-1], f12[2][0:-1]]
        puc2 = metrics.auc(f12[1], f12[0])
        fs2 = metrics.average_precision_score(cre1, sm2)
        cv2 = cov(cre1, sm2)[0][1]
        prs2, _ = per(cre1, sm2)
        sprs2, _ = sper(cre1, sm2)
        pb2, _ = pbs(cre1, (sm2 - np.min(sm2))/(np.max(sm2) - np.min(sm2)))
        sa2 = (sm2 - np.min(sm2))/(np.max(sm2) - np.min(sm2))
        # cv2 = cov(cre1, sa2)[0][1]

        h21, h22 = sm2[L], sm2[~L]
        hn = np.concatenate([h21, h22])
        hn = (hn - np.min(hn))/(np.max(hn) - np.min(hn))
        h21 = hn[0:len(h21)]
        h22 = hn[len(h21): len(h22)]
        md2 = abs(np.mean(h21) - np.mean(h22))

        sd = f12
        f1s2 = 0
        for u in range(len(sd[0])):
            f1s2 += 2*sd[0][u]*sd[1][u]/(sd[0][u] + sd[1][u])
        f1s2 /= len(sd[0])

        # sm44 = np.max(sm4) - sm4
        fpr4, tpr4, _ = metrics.roc_curve(cre1, np.max(sm4) - sm4)
        auc4 = metrics.roc_auc_score(cre1, np.max(sm4) - sm4)
        f14 = metrics.precision_recall_curve(cre1, np.max(sm4) - sm4)
        f14 = [f14[0][0:-1], f14[1][0:-1], f14[2][0:-1]]
        puc4 = metrics.auc(f14[1], f14[0])
        fs4 = metrics.average_precision_score(cre1, np.max(sm4) - sm4)
        cv4 = cov(cre1, np.max(sm4) - sm4)[0][1]
        prs4, _ = per(cre1, np.max(sm4) - sm4)
        sprs4, _ = sper(cre1, np.max(sm4) - sm4)
        sm44 = np.max(sm4) - sm4
        pb4, _ = pbs(cre1, (sm44 - np.min(sm44))/(np.max(sm44) - np.min(sm44)))
        sa4 = (sm44 - np.min(sm44))/(np.max(sm44) - np.min(sm44))
        # cv4 = cov(cre1, sa4)[0][1]

        h41, h42 = sm4[L], sm4[~L]
        hn = np.concatenate([h41, h42])
        hn = (hn - np.min(hn))/(np.max(hn) - np.min(hn))
        h41 = hn[0:len(h41)]
        h42 = hn[len(h41): len(h42)]
        md4 = abs(np.mean(h41) - np.mean(h42))

        sd = f14
        f1s4 = 0
        for u in range(len(sd[0])):
            f1s4 += 2*sd[0][u]*sd[1][u]/(sd[0][u] + sd[1][u])
        f1s4 /= len(sd[0])

        fpr5, tpr5, _ = metrics.roc_curve(cre1, sm5)
        auc5 = metrics.roc_auc_score(cre1, sm5)
        f15 = metrics.precision_recall_curve(cre1, sm5)
        f15 = [f15[0][0:-1], f15[1][0:-1], f15[2][0:-1]]
        puc5 = metrics.auc(f15[1], f15[0])
        fs5 = metrics.average_precision_score(cre1, sm5)
        cv5 = cov(cre1, sm5)[0][1]
        prs5, _ = per(cre1, sm5)
        sprs5, _ = sper(cre1, sm5)
        pb5, _ = pbs(cre1, (sm5 - np.min(sm5))/(np.max(sm5) - np.min(sm5)))
        sa5 = (sm5 - np.min(sm5))/(np.max(sm5) - np.min(sm5))
        # cv5 = cov(cre1, sa5)[0][1]

        h51, h52 = sm5[L], sm5[~L]
        hn = np.concatenate([h51, h52])
        hn = (hn - np.min(hn))/(np.max(hn) - np.min(hn))
        h51 = hn[0:len(h51)]
        h52 = hn[len(h51): len(h52)]
        md5 = abs(np.mean(h51) - np.mean(h52))

        sd = f15
        f1s5 = 0
        for u in range(len(sd[0])):
            f1s5 += 2*sd[0][u]*sd[1][u]/(sd[0][u] + sd[1][u])
        f1s5 /= len(sd[0])


        fpr7, tpr7, _ = metrics.roc_curve(cre1, np.max(sm7) - sm7)
        auc7 = metrics.roc_auc_score(cre1, np.max(sm7) - sm7)
        f17 = metrics.precision_recall_curve(cre1, np.max(sm7) - sm7)
        f17 = [f17[0][0:-1], f17[1][0:-1], f17[2][0:-1]]
        puc7 = metrics.auc(f17[1], f17[0])
        fs7 = metrics.average_precision_score(cre1, np.max(sm7) - sm7)
        cv7 = cov(cre1, np.max(sm7) - sm7)[0][1]
        prs7, _ = per(cre1, np.max(sm7) - sm7)
        sprs7, _ = sper(cre1, np.max(sm7) - sm7)
        sm77 = np.max(sm7) - sm7
        pb7, _ = pbs(cre1, (sm77 - np.min(sm77))/(np.max(sm77) - np.min(sm77)))
        sa7 = (sm77 - np.min(sm77))/(np.max(sm77) - np.min(sm77))
        # cv7 = cov(cre1, sa7)[0][1]

        h71, h72 = sm7[L], sm7[~L]
        hn = np.concatenate([h71, h72])
        hn = (hn - np.min(hn))/(np.max(hn) - np.min(hn))
        h71 = hn[0:len(h71)]
        h72 = hn[len(h71): len(h72)]
        md7 = abs(np.mean(h71) - np.mean(h72))

        sd = f17
        f1s7 = 0
        for u in range(len(sd[0])):
            f1s7 += 2*sd[0][u]*sd[1][u]/(sd[0][u] + sd[1][u])
        f1s7 /= len(sd[0])

        fpr8, tpr8, _ = metrics.roc_curve(cre1, sm8)
        auc8 = metrics.roc_auc_score(cre1, sm8)
        f18 = metrics.precision_recall_curve(cre1, sm8)
        f18 = [f18[0][0:-1], f18[1][0:-1], f18[2][0:-1]]
        puc8 = metrics.auc(f18[1], f18[0])
        fs8 = metrics.average_precision_score(cre1, sm8)
        cv8 = cov(cre1, sm8)[0][1]
        prs8, _ = per(cre1, sm8)
        sprs8, _ = sper(cre1, sm8)
        pb8, _ = pbs(cre1, (sm8 - np.min(sm8))/(np.max(sm8) - np.min(sm8)))
        sa8 = (sm8 - np.min(sm8))/(np.max(sm8) - np.min(sm8))
        # cv8 = cov(cre1, sa8)[0][1]

        h81, h82 = sm8[L], sm8[~L]
        hn = np.concatenate([h81, h82])
        hn = (hn - np.min(hn))/(np.max(hn) - np.min(hn))
        h81 = hn[0:len(h81)]
        h82 = hn[len(h81): len(h82)]
        md8 = abs(np.mean(h81) - np.mean(h82))

        sd = f18
        f1s8 = 0
        for u in range(len(sd[0])):
            f1s8 += 2*sd[0][u]*sd[1][u]/(sd[0][u] + sd[1][u])
        f1s8 /= len(sd[0])

        lw = 2

        plt.figure()
        plt.subplot(1, 6, 1)
        plt.hist(h11, density=True, alpha = 0.5, color='r')
        plt.hist(h12, density=True, alpha = 0.5, color='b')
        # plt.grid(linestyle='dotted')
        plt.ylabel("FREQUENCY")
        plt.title('MC-DROPOUT')
        # plt.xlabel('DETECTIONS')
        # plt.ylabel('ERROR CLASS MEAN DIFF')
        plt.subplot(1, 6, 2)
        plt.hist(h21, density=True, alpha = 0.5, color='r')
        plt.hist(h22, density=True, alpha = 0.5, color='b')
        # plt.grid(linestyle='dotted')
        plt.title('MC-DROPOUT-LL')
        plt.yticks([])
        # plt.xlabel('NOISE SEVERITY')
        # plt.ylabel('ERROR CLASS MEAN DIFF')
        plt.subplot(1, 6, 3)
        plt.hist(h41, density=True, alpha = 0.5, color='r')
        plt.hist(h42, density=True, alpha = 0.5, color='b')
        # plt.grid(linestyle='dotted')
        plt.title('SVI-LL')
        plt.yticks([])
        # plt.xlabel('NOISE SEVERITY')
        # plt.ylabel('ERROR CLASS MEAN DIFF')
        plt.subplot(1, 6, 4)
        plt.hist(h51, density=True, alpha = 0.5, color='r')
        plt.hist(h52, density=True, alpha = 0.5, color='b')
        # plt.grid(linestyle='dotted')
        plt.yticks([])
        plt.title('QIPF')
        # plt.xlabel('NOISE SEVERITY')
        # plt.ylabel('ERROR CLASS MEAN DIFF')
        plt.subplot(1, 6, 5)
        plt.hist(h71, density=True, alpha = 0.5, color='r')
        plt.hist(h72, density=True, alpha = 0.5, color='b')
        # plt.grid(linestyle='dotted')
        plt.yticks([])
        plt.title('SVI')
        # plt.xlabel('NOISE SEVERITY')
        # plt.ylabel('ERROR CLASS MEAN DIFF')
        plt.subplot(1, 6, 6)
        plt.hist(h81, density=True, alpha = 0.5, color='r')
        plt.hist(h82, density=True, alpha = 0.5, color='b')
        # plt.grid(linestyle='dotted')
        plt.yticks([])
        plt.title('ENSEMBLE')
        plt.xlabel("UNCERTAINTY ESTIMATES (NORMALIZED): BLUE = CORRECT PREDICTIONS, PINK = WRONG PREDICTIONS")
        # plt.xlabel('NOISE SEVERITY')
        # plt.ylabel('ERROR CLASS MEAN DIFF')

        plt.figure()
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        plt.plot(fpr1, tpr1, label="MC_DROP (AUC = %0.2f)" % auc1)
        plt.plot(fpr2, tpr2, linestyle='-.', label="MC_DROP_LL (AUC = %0.2f)" % auc2)
        plt.plot(fpr4, tpr4, linestyle=':', label="SVI_LL (AUC = %0.2f)" % auc4)
        plt.plot(fpr5, tpr5, linestyle='-', label="QIPF (AUC = %0.2f)" % auc5)
        plt.plot(fpr7, tpr7, linestyle=':', label="SVI (AUC = %0.2f)" % auc7)
        plt.plot(fpr8, tpr8, linestyle='--', label="ENSEMBLE (AUC = %0.2f)" % auc8)
        plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
        plt.legend(loc=4)
        plt.grid(linestyle='dotted')
        plt.xlabel('FALSE POSITIVE RATE')
        plt.ylabel('TRUE POSITIVE RATE')
        plt.show()

        plt.figure()
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        plt.plot(f11[1], f11[0], label="MC_DROP (AUC = %0.2f)" % puc1)
        plt.plot(f12[1], f12[0], linestyle='-.', label="MC_DROP_LL (AUC = %0.2f)" % puc2)
        plt.plot(f14[1], f14[0], linestyle=':', label="SVI_LL (AUC = %0.2f)" % puc4)
        plt.plot(f15[1], f15[0], linestyle='-', label="QIPF (AUC = %0.2f)" % puc5)
        plt.plot(f17[1], f17[0], linestyle=':', label="SVI (AUC = %0.2f)" % puc7)
        plt.plot(f18[1], f18[0], linestyle='--', label="ENSEMBLE (AUC = %0.2f)" % puc8)
        # plt.plot([0, 1], [1, 0], color='black', lw=lw, linestyle='--')
        plt.legend(loc=4)
        plt.grid(linestyle='dotted')
        plt.ylabel('PRECISION')
        plt.xlabel('RECALL')
        plt.show()

        return auc1, auc2, auc4, auc5, auc7, auc8, f11, f12, f14, f15, f17, f18, fs1, fs2, fs4, \
               fs5, fs7, fs8, cv1, cv2, cv4, cv5, cv7, cv8, prs1, prs2, prs4, prs5, \
               prs7, prs8, sprs1, sprs2, sprs4, sprs5, sprs7, sprs8, pb1, pb2, pb4, pb5, \
               pb7, pb8, f1s1, f1s2, f1s4, f1s5, f1s7, f1s8, puc1, puc2, puc4, puc5, puc7, \
               puc8, md1, md2, md4, md5, md7, md8

    def brier(yt, yp):
        # from sklearn.metrics import brier_score_loss as br
        # return br(yt, yp)
        from uncertainty_metrics.tensorflow.scoring_rules import brier_score
        return brier_score(yt, yp)

    def ece(yt, yp):
        import uncertainty_metrics.numpy as um
        ec = um.ece(yt, yp)
        return ec

    def nll(yt, yp):
        from scipy import special
        nl = -special.xlogy(yt, yp) - special.xlogy(1 - yt, 1 -yp)
        return nl
