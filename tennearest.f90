! Created by  on 06/04/2022.

PROGRAM main
    implicit none
    !-- Variable definion
    INTEGER :: n, i, j, father_id
    ! TODO : Make the norme automatically updated by the program main.py and also the taille_cat
    REAL, PARAMETER :: Space_norme = 10.6 ! distance 'max' between two event, in km
    REAL, PARAMETER :: Time_norme = 28.7 * 24 * 60 * 60 ! time 'max' between two event, in sec
    INTEGER, PARAMETER :: taille_cat = 1341  !40171 !9073 !33119 ! true cat lenght ATTENTION must be superior or egal to the argument "j"
    DOUBLE PRECISION, DIMENSION(taille_cat) :: Rji, Tji, Tj, Rj, Dji, Dj, F_id
    LOGICAL, DIMENSION(taille_cat) :: mask
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
    !-- read args
    CALL getarg(1, line1)
    read(line1, '(i7)') j

    !-- read cat

    open(unit = 18, file = 'output/cat_formated.csv', form = 'formatted',status='unknown')

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

    ! Find nearest father based on time and distance
    Rji = 9999;Tji = 9999;Dji = 9999
    DO i = 1, j - 1 ! attention i<j is mendatory
        Rji(i) = haversine(cat(i)%lon, cat(i)%lat, cat(j)%lon, cat(j)%lat)
        Tji(i) = ABS(cat(i)%time - cat(j)%time) * 12 * 31 * 24 * 60 * 60
        Dji(i) = sqrt((Rji(i) / Space_norme)**2 + (Tji(i) / Time_norme)**2)
    END DO

    mask = .TRUE.
    Rj = 9999;Tj = 9999;Dj = 9999;F_id=0
    DO i = 1, 10
        father_id = MINLOC(Dji, 1, mask)

        F_id(i) = father_id
        Rj(i) = Rji(father_id)
        Tj(i) = Tji(father_id)
        Dj(i) = Dji(father_id)
        mask(father_id) = .FALSE.
    END DO

    DO i = 1, 10
        n = F_id(i)
        PRINT *, F_id(i), Rj(i), Tj(i) , Dj(i)
    END DO

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

