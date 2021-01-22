PROGRAM MAIN
    USE NOGWDNN_MOD, ONLY: NN, NN_TL, NN_AD
    USE YONOGWDNN
    USE PARKIND1, ONLY: NNP, DOUB

    IMPLICIT NONE
    
    REAL(NNP) :: X1(NINP), X2(NINP), Y1(NOUT), Y2(NOUT), Y3(NOUT), DX(NINP), DY(NOUT)
    REAL(NNP) :: DX1(NINP), DX2(NINP), DY1(NOUT), DY2(NOUT)
    REAL(DOUB) :: LHS, RHS, DIFF
    INTEGER :: I
    
    ALLOCATE(INPUT_B(NWIDTH))
    ALLOCATE(OUTPUT_B(NOUT))
    ALLOCATE(INPUT(NWIDTH, NINP))
    ALLOCATE(OUTPUT(NOUT, NWIDTH))
    ALLOCATE(HIDDEN_B(NWIDTH, NHIDDEN))
    ALLOCATE(HIDDEN(NWIDTH, NWIDTH, NHIDDEN))
    
    CALL INITIALIZE1(INPUT_B)
    CALL INITIALIZE1(OUTPUT_B)
    CALL INITIALIZE2(INPUT)
    CALL INITIALIZE2(OUTPUT)
    CALL INITIALIZE2(HIDDEN_B)
    CALL INITIALIZE3(HIDDEN)
    
    CALL INITIALIZE1(X1)
    
    DO I = 0, 7
        CALL INITIALIZE1(DX)
        DX = DX*10.0_NNP**(-REAL(I, NNP))
        X2 = X1 + DX

        WRITE (*,'(A)') "====================================================="
        WRITE (*,'(A,ES7.1)') "Size of perturbation = ", DX(1)
        
        CALL NN(X1, Y1)
        CALL NN(X2, Y2)
        CALL NN_TL(X1, DX, Y3, DY)
        
        WRITE (*,'(A,10F21.17)') "LHS", Y2 - Y1
        WRITE (*,'(A,5F21.17)') "RHS", DY
        DIFF = ABS(Y2(1) - Y1(1) - DY(1))
        WRITE (*,'(A,I3)') "Difference of 1st element around digit number ", NINT(ABS(LOG10(DIFF)))
        WRITE (*,'(A)') "Nonlinear trajectory comparison (should be identical):"
        WRITE (*,'(5F21.17)') Y1
        WRITE (*,'(5F21.17)') Y3
    END DO

    ! Adjoint test
    CALL INITIALIZE1(DX1)
    CALL INITIALIZE1(DY2)
    DX1 = DX1*10.0_NNP**(-REAL(1, NNP))
    DY2 = DY2*10.0_NNP**(-REAL(1, NNP))

    CALL NN_TL(X1, DX1, Y1, DY1)
    CALL NN_AD(X1, DY2, DX2)

    WRITE (*,'(A)') "====================================================="
    WRITE (*,'(A)') ""
    WRITE (*,'(A)') "====================================================="
    WRITE (*,'(A)') "Adjoint test"
    WRITE (*,'(A)') "The following two numbers must be as similar as possible"

    LHS = DOT_PRODUCT(DY1, DY2)
    RHS = DOT_PRODUCT(DX1, DX2)
    WRITE (*,'(A,F21.17)') 'LHS', LHS
    WRITE (*,'(A,F21.17)') 'RHS', RHS
    DIFF = ABS(LHS - RHS)
    WRITE (*,'(A,I3)') "Difference of 1st element around digit number ", NINT(ABS(LOG10(DIFF)))

    
CONTAINS
    SUBROUTINE INITIALIZE1(X)
        REAL(NNP), INTENT(OUT) :: X(:)
        
        INTEGER :: I
        
        DO I = 1, SIZE(X)
            X(I) = RANDN(0.0_NNP, 1.0_NNP)
        END DO
    END SUBROUTINE INITIALIZE1

    SUBROUTINE INITIALIZE2(X)
        REAL(NNP), INTENT(OUT) :: X(:,:)
        
        INTEGER :: I, J
        
        DO I = 1, SIZE(X, 1)
            DO J = 1, SIZE(X, 2)
                X(I, J) = RANDN(0.0_NNP, 1.0_NNP)
            END DO
        END DO
    END SUBROUTINE INITIALIZE2

    SUBROUTINE INITIALIZE3(X)
        REAL(NNP), INTENT(OUT) :: X(:,:,:)
        
        INTEGER :: I, J, K
        
        DO I = 1, SIZE(X, 1)
            DO J = 1, SIZE(X, 2)
                DO K = 1, SIZE(X, 3)
                    X(I, J, K) = RANDN(0.0_NNP, 1.0_NNP)
                END DO
            END DO
        END DO
    END SUBROUTINE INITIALIZE3

    FUNCTION RANDN(MEAN, STDEV)
        REAL(NNP), INTENT(IN) :: MEAN, STDEV
        REAL(NNP) :: U, V, RANDN
        REAL(NNP) :: RAND(2)

        CALL RANDOM_NUMBER(RAND)

        ! Box-Muller method
        U = (-2.0_NNP * LOG(RAND(1))) ** 0.5_NNP
        V =   2.0_NNP * 6.28318530718_NNP * RAND(2)
        RANDN = MEAN + STDEV * U * SIN(V)
    END FUNCTION RANDN
END PROGRAM MAIN
