! Created by  on 03/05/2022.
PROGRAM main
    implicit none

    include 'output/fvar.f'
    !-- Variable definion
    INTEGER :: n, i, j, father_id, father_id2 , jj
    DOUBLE PRECISION, DIMENSION(taille_cat) :: Rji, Tji, Tj, Rj, AR, AT
    DOUBLE PRECISION :: T, R ,a
    LOGICAL, DIMENSION(taille_cat) :: mask, mask2
    CHARACTER (len = 10) :: line1

    Type catalogue
        INTEGER :: ID
        REAL :: lon
        REAL :: lat
        DOUBLE PRECISION :: time
        REAL :: mag
        REAL :: depth
        CHARACTER (len = 30) :: time_name
    End Type catalogue

    TYPE(catalogue), DIMENSION(taille_cat) :: cat

    !-- read cat

    open(unit = 18, file = 'output/cat_formated.csv', form = 'formatted')

    DO i = 0, taille_cat
        IF (i > 0) THEN
            read(18, *) cat(i)
        ELSE
            read(18, *)
        END IF
    END DO

    close(18)

    !    DO i = 1, taille_cat
    !        print*, cat(i)
    !    END DO
    DO j = 1, taille_cat
        ! Find nearest father based on time and distance
        Rji = 9999;Tji = 9999
        DO i = 1, j - 1 ! attention i<j is mendatory
            Rji(i) = haversine(cat(i)%lon, cat(i)%lat, cat(j)%lon, cat(j)%lat)
            Tji(i) = ABS(cat(i)%time - cat(j)%time)*365
        END DO

        mask = .TRUE.
        mask2 = .TRUE.
        Rj = 9999;Tj = 9999
        DO i = 1, MIN(100,taille_cat)
            father_id = MINLOC(Rji, 1, mask)
            father_id2 = MINLOC(Tji, 1, mask2)

            Rj(i) = Rji(father_id)
            Tj(i) = Tji(father_id2)
            mask(father_id) = .FALSE.
            mask2(father_id2) = .FALSE.
        END DO

        R=0;T=0
        DO i = 1, MIN(100,taille_cat)
            R = Rj(i) + R
            T = Tj(i)  + T
        END DO

        AR(j) = R/100
        AT(j) = T/100
    END DO

    do jj = 2, taille_cat
        a = AR(jj)
        do i = jj - 1, 1, -1
            if (AR(i)<=a) goto 11
            AR(i + 1) = AR(i)
        end do
        i = 0
        11  AR(i + 1) = a
    end do

    do jj = 2, taille_cat
        a = AT(jj)
        do i = jj - 1, 1, -1
            if (AT(i)<=a) goto 10
            AT(i + 1) = AT(i)
        end do
        i = 0
        10  AT(i + 1) = a
    end do

    print*, 'Norme spatiale (km) ', AR(3*taille_cat/4), ' Norme temporelle (day)', AT(3*taille_cat/4)
    ! define fonction
CONTAINS
    DOUBLE PRECISION FUNCTION haversine(lon11, lat11, lon22, lat22)
        !Calculate the great circle distance in kilometers between two points
        !on the earth (specified in decimal degrees)

        REAL lon1, lon2, lat1, lat2, dlon, dlat, lat11, lon11, lat22, lon22
        DOUBLE PRECISION  a, b, c, r
        ! convert decimal degrees to radians
        lon1 = lon11 * 3.1415 / 180
        lat1 = lat11 * 3.1415 / 180
        lon2 = lon22 * 3.1415 / 180
        lat2 = lat22 * 3.1415 / 180

        ! haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * asin(sqrt(a))
        r = 6371 ! Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
        haversine = c * r
    END FUNCTION haversine
END PROGRAM main

