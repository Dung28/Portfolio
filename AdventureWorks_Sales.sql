---- DIM CUSTOMER
SELECT CustomerID, BusinessEntityID, StoreID, Title, FirstName, MiddleName, LastName, max([4]) as StateProvinceID,
  max([2]) as HomeAddress,max([3]) as City1, max([5]) as ShippingAddress, max([6]) as City2 
  FROM
  (SELECT G.CustomerID, A.BusinessEntityID ,a.Title, g.StoreID, A.FirstName, A.MiddleName, A.LastName, C.EmailAddress, D.PhoneNumber,
  B.AddressID, B.AddressTypeID, B.AddressTypeID + '1' as AddressTypeID1, B.AddressTypeID + '2' as AddressTypeID2,  E.AddressLine1, E.City, h.StateProvinceID
  FROM Person.Person A JOIN Person.BusinessEntityAddress B ON A.BusinessEntityID = B.BusinessEntityID
  JOIN Person.EmailAddress C ON A.BusinessEntityID = C.BusinessEntityID
  JOIN Person.PersonPhone D ON A.BusinessEntityID = D.BusinessEntityID
  JOIN Person.Address E ON B.AddressID = E.AddressID
  join Sales.Customer G on g.PersonID = a.BusinessEntityID 
  join Person.Address H on h.AddressID = b.AddressID 
 
  WHERE A.PersonType = 'IN' ) TableSource
  PIVOT
  (
  MAX (AddressLine1)
  for AddressTypeID
  IN([2], [5]) )Pvt1
  PIVOT
  (
  MAX (City)
  for AddressTypeID1
  IN([3], [6])
  ) Pvt2
 PIVOT
  (
  MAX (StateProvinceID)
  for AddressTypeID2
  IN([4], [7])
  ) Pvt3
 GROUP BY CustomerID,Title, StoreID, FirstName, MiddleName, LastName, EmailAddress, PhoneNumber, BusinessEntityID

 /*SELECT G.CustomerID, A.BusinessEntityID ,a.Title, g.StoreID, A.FirstName, A.MiddleName, A.LastName, C.EmailAddress, D.PhoneNumber,
  B.AddressID,  E.AddressLine1 Adress, E.City into test.dbo.[check]
  FROM Person.Person A JOIN Person.BusinessEntityAddress B ON A.BusinessEntityID = B.BusinessEntityID
  JOIN Person.EmailAddress C ON A.BusinessEntityID = C.BusinessEntityID
  JOIN Person.PersonPhone D ON A.BusinessEntityID = D.BusinessEntityID
  JOIN Person.Address E ON B.AddressID = E.AddressID
  join Sales.Customer G on g.PersonID = a.BusinessEntityID 
  join Person.Address H on h.AddressID = b.AddressID 
  where A.PersonType ='IN' and AddressTypeID= 2*/


---- DIM TERRITORY
SELECT TerritoryID, Name, [Group]
FROM Sales.SalesTerritory

---- DIMSTORE
SELECT BusinessEntityID StoreID,  Name StoreName, SalesPersonID, [3] AS AddressLine1,[5] AS Address2, City

FROM
(select A.BusinessEntityID,  A.Name, A.SalesPersonID, B.AddressTypeID, C.AddressLine1, C.City, C.PostalCode
FROM SALES.STORE A
JOIN PERSON.BUSINESSENTITYADDRESS B ON a.BusinessEntityID = B.BusinessEntityID
JOIN PERSON.Address C ON C.AddressID = B.AddressID) TABLESOURCE
Pivot 
(

MAX(AddressLine1)
FOR AddressTypeID
IN ([3] , [5])
) AS PIVOTTABLE

----- DimSalePerson
Select A. BusinessEntityID SalerID, B.FirstName, B.LastName 
from Sales.SalesPerson A join Person.Person B on A.BusinessEntityID = B.BusinessEntityID


---- DimLocation

select A.BusinessEntityID, D.AddressID, G.TerritoryID, G.Name Territory, G.[Group], G.SalesYTD, G.SalesLastYear, E.StateProvinceCode, E.Name StateProvince,
F.CountryRegionCode, F.Name CountryRegion 

from Person.BusinessEntityAddress A left join Sales.Store B on a.BusinessEntityID = b.BusinessEntityID
	 left join Sales.Customer C on A.BusinessEntityID = C. PersonID
	 left join Person.Person Q on Q.BusinessEntityID = C.PersonID 
	 Join Person.Address D on D.AddressID = A.AddressID
	 join Person.StateProvince E on E.StateProvinceID = D.StateProvinceID
	 join Person.CountryRegion F on F.CountryRegionCode = E.CountryRegionCode
	 join Sales.SalesTerritory G on G.TerritoryID = E.TerritoryID
	 where b.BusinessEntityID is not null or Q.PersonType = 'IN'
	

select *
from Sales.SalesTerritory

----- DimProductCategory
select ProductCategoryID, Name 
from Production.ProductCategory


----DimSubcategory
select ProductSubcategoryID, ProductCategoryID, Name
from Production.ProductSubcategory

---DimPromotion
select SpecialOfferID, Description, DiscountPct, Type, Category, 
FORMAT( StartDate,'yyyyMMdd') StartDate, 
FORMAT( EndDate,'yyyyMMdd') EndDate,
MaxQty, MinQty from Sales.SpecialOffer


--DimProduct
Select ProductID, Name, StandardCost, 
Color, ListPrice, Size, Weight, ProductLine, ProductSubcategoryID,
cast(FORMAT(SellStartDate,'yyyyMMdd') as int) SellStartDate,
cast(FORMAT( SellEndDate,'yyyyMMdd') as int) SellEndDate 
from Production.Product 
where ProductSubcategoryID is not null


--FactSales
select c.CustomerID, 
ProductID, 
SpecialOfferID,
a.SalesOrderID, 
e.SalesReasonID,
FORMAT(OrderDate,'yyyyMMdd') OrderDate, 
FORMAT( DueDate,'yyyyMMdd') DueDate,
FORMAT( ShipDate,'yyyyMMdd') ShipDate, 
OrderQty, 
UnitPrice, 
UnitPriceDiscount, 
LineTotal, 
LineTotal*0.1 as Tax
from Sales.SalesOrderHeader a join Sales.SalesOrderDetail b 
on a.SalesOrderID= b.SalesOrderID
join Sales.Customer c on a.CustomerID = c.CustomerID
join Person.Person d on c.PersonID=d.BusinessEntityID
join Sales.SalesOrderHeaderSalesReason e on a.SalesOrderID = e.SalesOrderID
where d.PersonType='IN'

--UPDATE [dbo].[FactSales]
--SET [LineTotal] =?
--where [CustomerKey] =? 
--and [ProductKey] =?
--and [SpecialOrderKey] =?
--and [SalesReasonKey] =?

--DimSalesReason
select Name, SalesReasonID, ReasonType  from Sales.SalesReason

--FactSalesQuota
select BusinessEntityID SalespersonID, 
FORMAT(QuotaDate,'yyyyMMdd') QuotaDate, SalesQuota 
from sales.SalesPersonQuotaHistory

------UPDATE [dbo].[FactSalesQuota]
--SET [SalesQuota] =?
--WHERE [SalesPersonKey] =? 


---DimTime
SELECT format(pk_date, 'yyyyMMdd') TimeKey,
PK_Date,
year(pk_date) [Year], MONTH(pk_date) [Month], day(pk_date) [Day],
DATENAME(weekday, PK_Date) [DayOfWeek], 
DATEPART(QUARTER, PK_Date) [Quarter]
into DimTime
FROM TIME



----------------------- Test
select B.BusinessEntityID, B.Name, A.AddressID, C.AddressLine1, c.City, d.Name StateProvince, e.Name CountryRegion, 
G.Name Territory, G.[Group], C.AddressLine2

from Person.BusinessEntityAddress A join Sales.Store B on a.BusinessEntityID = b.BusinessEntityID
join Person.Address C on c.AddressID = a.AddressID
join Person.StateProvince D on D.StateProvinceID = C.StateProvinceID
join Person.CountryRegion E on e.CountryRegionCode = d.CountryRegionCode
join Sales.SalesTerritory G on G.TerritoryID = D.TerritoryID