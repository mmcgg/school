#!/usr/bin/env python
import numpy as np
np.set_printoptions(precision=1)

G = 6.674e-11 #m^3 kg^-1 s^-2

def calculate_mass_ratio(principal_mass, other_masses):
    return principal_mass/np.sum(other_masses)

def calculate_angular_momentum_of_satellites(principal_mass,masses, distances):
    totalAngularMomentum = 0
    numberOfSatellites = len(masses)-1
    
    for i in xrange(0,numberOfSatellites):
        # Using equation 2.26 and assume e=0 (a circular orbit)
        totalAngularMomentum += masses[i]*np.sqrt(G*principal_mass*distances[i])
        
    return totalAngularMomentum

AU2km = 1.496e8
# Masses in kg
massMercury = .3302e24
massVenus = 4.8685e24
massEarth = 5.9736e24
massMars = .64185e24
massJupiter = 1898.6e24
massSaturn = 568.46e24
massUranus = 86.832e24
massNeptune = 102.43e24
massSun = massEarth*332900 # From wikipedia ("Solar System")

# Some needed radii in km
radiusSun = 6.955e5
radiusJupiter = 71492
radiusSaturn = 60268
radiusUranus = 25559

# Some needed inertias in kg m^2
inertiaSun = .059*massSun*radiusSun**2.0
inertiaJupiter = .254*massJupiter*radiusJupiter**2.0
inertiaSaturn = .21*massSaturn*radiusSaturn**2.0
inertiaUranus = .23*massUranus*radiusUranus**2.0

# Some needed rotational periods in s
rotationalPeriodSun = 25.4*24*60*60
rotationalPeriodJupiter = 9*60*60 + 55*60 + 29.71
rotationalPeriodSaturn = 10*60*60 + 32*60 + 35
rotationalPeriodUranus = 17*60*60 + 14*60

# distances in km
aMercury = .3871*AU2km
aVenus = .7233*AU2km
aEarth = 1.0*AU2km
aMars = 1.5237*AU2km
aJupiter = 5.203*AU2km
aSaturn = 9.543*AU2km
aUranus = 19.192*AU2km
aNeptune = 30.069*AU2km

sunSatelliteMassList = [massMercury,massVenus,massEarth,massMars,massJupiter,massSaturn,massUranus,massNeptune]
sunSatelliteDistanceList = [aMercury,aVenus,aEarth,aMars,aJupiter,aSaturn,aUranus,aNeptune]

# Masses and semi-major axes of Jupiter's satellites
massIo = 893.3e20
massEuropa = 479.7e20
massGanymede = 1482e20
massHimalia = .042e20
aIo = 421.77e3
aEuropa = 671.08e3
aGanymede = 1070.4e3
aHimalia = 11460e3
jupiterSatelliteMassList = [massIo,massEuropa,massGanymede,massHimalia]
jupiterSatelliteDistanceList = [aIo,aEuropa,aGanymede,aHimalia]

# Masses and semi-major axes of Saturn's satellites
massJanus = .019e20
massMimas = .38e20
massEnceladus = .65e20
massTethys = 6.27e20
massDione = 11.0e20
massRhea = 23.1e20
massTitan = 1345.7e20
massHyperion = .056e20
massIapetus = 18.1e20
massPhoebe = .083e20
aJanus = 151.47e3
aMimas = 185.52e3
aEnceladus = 238.02e3
aTethys = 294.66e3
aDione = 377.71e3
aRhea = 527.04e3
aTitan = 1221.85e3
aHyperion = 1481.1e3
aIapetus = 3561.3e3
aPhoebe = 12952.0e3

saturnSatelliteMassList = [massJanus,massMimas,massEnceladus,massTethys,massDione,massRhea,massTitan,massHyperion,massIapetus,massPhoebe]
saturnSatelliteDistanceList = [aJanus,aMimas,aEnceladus,aTethys,aDione,aRhea,aTitan,aHyperion,aIapetus,aPhoebe]

# Masses and semi-major axes of Uranus's satellites
massMiranda = .659e20
massAriel = 13.53e20
massUmbriel = 11.72e20
massTitania = 35.27e20
massOberon = 30.14e20
aMiranda = 129.8e3
aAriel = 191.2e3
aUmbriel = 266.0e3
aTitania = 435.8e3
aOberon = 582.6e3
uranusSatelliteMassList = [massMiranda,massAriel,massUmbriel,massTitania,massOberon]
uranusSatelliteDistanceList = [aMiranda,aAriel,aUmbriel,aTitania,aOberon]



print "Part A:"
print "Sun to planets mass ratio: ", calculate_mass_ratio(massSun,sunSatelliteMassList)
print "Jupiter to moons mass ratio: ", calculate_mass_ratio(massJupiter,jupiterSatelliteMassList)
print "Saturn to moons mass ratio: ", calculate_mass_ratio(massSaturn,saturnSatelliteMassList)
print "Uranus to moons mass ratio: ", calculate_mass_ratio(massUranus,uranusSatelliteMassList)

print "\n\n"

print "Part B:"
angularMomentumSun = inertiaSun*(1.0/rotationalPeriodSun)
angularMomentumPlanets = calculate_angular_momentum_of_satellites(massSun,sunSatelliteMassList,sunSatelliteDistanceList)
# print "Angular momentum of the sun: ", angularMomentumSun
# print "Angular momentum of the sun's satellites: ",angularMomentumPlanets
print "Ratio of sun to planets angular momenta: ", angularMomentumSun/angularMomentumPlanets

print "\n\n"

print "Part C:"
angularMomentumJupiter = inertiaJupiter*(1.0/rotationalPeriodJupiter)
angularMomentumJupiterSatellites = calculate_angular_momentum_of_satellites(massJupiter,jupiterSatelliteMassList,jupiterSatelliteDistanceList)
# print "Angular momentum of Jupiter: ", angularMomentumJupiter
# print "Angular momentum of the Jupiter's satellites: ",angularMomentumJupiterSatellites
print "Ratio of Jupiter to satellites angular momenta: ", angularMomentumSun/angularMomentumJupiterSatellites

angularMomentumSaturn = inertiaSaturn*(1.0/rotationalPeriodSaturn)
angularMomentumSaturnSatellites = calculate_angular_momentum_of_satellites(massSaturn,saturnSatelliteMassList,saturnSatelliteDistanceList)
# print "Angular momentum of Saturn: ", angularMomentumSaturn
# print "Angular momentum of the Saturn's satellites: ",angularMomentumSaturnSatellites
print "Ratio of Saturn to satellites angular momenta: ", angularMomentumSun/angularMomentumSaturnSatellites

angularMomentumUranus = inertiaUranus*(1.0/rotationalPeriodUranus)
angularMomentumUranusSatellites = calculate_angular_momentum_of_satellites(massUranus,uranusSatelliteMassList,uranusSatelliteDistanceList)
# print "Angular momentum of Uranus: ", angularMomentumUranus
# print "Angular momentum of the Uranus's satellites: ",angularMomentumUranusSatellites
print "Ratio of Uranus to satellites angular momenta: ", angularMomentumSun/angularMomentumUranusSatellites

print "\n\n"

print "Part D"
print "Mercury semi-major axis in solar radii: ",aMercury/radiusSun
print "Venus semi-major axis in solar radii: ",aVenus/radiusSun
print "Earth semi-major axis in solar radii: ",aEarth/radiusSun
print "Mars semi-major axis in solar radii: ",aMars/radiusSun
print "Jupiter semi-major axis in solar radii: ",aJupiter/radiusSun
print "Saturn semi-major axis in solar radii: ",aSaturn/radiusSun
print "Uranus semi-major axis in solar radii: ",aUranus/radiusSun
print "Neptune semi-major axis in solar radii: ",aNeptune/radiusSun

print "\n\n"
aIo = 421.77e3
aEuropa = 671.08e3
aGanymede = 1070.4e3
aHimalia = 11460e3
print "Io semi-major axis in Jupiter radii: ",aIo/radiusJupiter
print "Europa semi-major axis in solar radii: ",aEuropa/radiusJupiter
print "Ganymede semi-major axis in solar radii: ",aGanymede/radiusJupiter
print "Himalia semi-major axis in solar radii: ",aHimalia/radiusJupiter



